import React, { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import type {
  UserResponse,
  OrgMembershipResponse,
  LoginRequest,
  RegisterRequest,
} from '../types';
import {
  getAuthToken,
  setAuthToken,
  getSelectedOrgId,
  setSelectedOrgId,
  clearAuthSession,
  setUnauthorizedHandler,
  loginUser,
  registerUser,
  logoutUser,
  fetchCurrentUser,
} from '../services/api';

export interface AuthContextType {
  user: UserResponse | null;
  organizations: OrgMembershipResponse[];
  currentOrg: OrgMembershipResponse | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  authError: string | null;
  login: (credentials: LoginRequest) => Promise<void>;
  register: (data: RegisterRequest) => Promise<void>;
  logout: () => Promise<void>;
  switchOrganization: (orgId: string) => void;
  restoreSession: () => Promise<void>;
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

export interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [organizations, setOrganizations] = useState<OrgMembershipResponse[]>([]);
  const [currentOrg, setCurrentOrg] = useState<OrgMembershipResponse | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [authError, setAuthError] = useState<string | null>(null);

  /**
   * Reset all authentication state and clear stored credentials.
   * Awaits server-side token revocation while guaranteeing local session clearing in finally.
   */
  const logout = useCallback(async (): Promise<void> => {
    try {
      await logoutUser();
    } catch {
      // Ignore network errors or aborted calls
    } finally {
      clearAuthSession();
      setUser(null);
      setOrganizations([]);
      setCurrentOrg(null);
      setIsAuthenticated(false);
      setAuthError(null);
    }
  }, []);

  /**
   * Resolve active organization selection based on verified memberships:
   * 1. Exactly 1 membership -> auto-select.
   * 2. Multiple memberships -> keep previously stored selection only if it matches
   *    one of the active returned memberships; otherwise clear stored selection.
   * 3. 0 memberships -> clear selection.
   */
  const resolveOrganizationSelection = useCallback(
    (memberships: OrgMembershipResponse[]): OrgMembershipResponse | null => {
      if (!memberships || memberships.length === 0) {
        setSelectedOrgId(null);
        return null;
      }

      if (memberships.length === 1) {
        const singleOrg = memberships[0];
        setSelectedOrgId(singleOrg.organization_id);
        return singleOrg;
      }

      // Multiple memberships: check if previously stored selection is valid
      const storedOrgId = getSelectedOrgId();
      if (storedOrgId) {
        const matchingOrg = memberships.find((m) => m.organization_id === storedOrgId);
        if (matchingOrg) {
          setSelectedOrgId(matchingOrg.organization_id);
          return matchingOrg;
        }
      }

      // No valid stored selection: clear storage and require explicit selection
      setSelectedOrgId(null);
      return null;
    },
    []
  );

  /**
   * Restore user session from stored access token.
   */
  const restoreSession = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    setAuthError(null);

    const token = getAuthToken();
    if (!token || !token.trim()) {
      clearAuthSession();
      setUser(null);
      setOrganizations([]);
      setCurrentOrg(null);
      setIsAuthenticated(false);
      setIsLoading(false);
      return;
    }

    try {
      const meResponse = await fetchCurrentUser();
      const loadedUser = meResponse.user;
      const loadedOrgs = meResponse.organizations || [];

      if (!loadedUser || !loadedUser.is_active) {
        logout();
        setAuthError('Account is inactive or disabled.');
        setIsLoading(false);
        return;
      }

      const selected = resolveOrganizationSelection(loadedOrgs);

      setUser(loadedUser);
      setOrganizations(loadedOrgs);
      setCurrentOrg(selected);
      setIsAuthenticated(true);
      setAuthError(null);
    } catch (err: unknown) {
      // Invalid/expired token or network failure
      logout();
      const message = err instanceof Error ? err.message : 'Session verification failed';
      setAuthError(message);
    } finally {
      setIsLoading(false);
    }
  }, [logout, resolveOrganizationSelection]);

  /**
   * Log in user with credentials and restore session.
   */
  const login = useCallback(
    async (credentials: LoginRequest): Promise<void> => {
      setIsLoading(true);
      setAuthError(null);
      try {
        const tokenResponse = await loginUser(credentials);
        setAuthToken(tokenResponse.access_token);
        await restoreSession();
      } catch (err: unknown) {
        logout();
        const message = err instanceof Error ? err.message : 'Login failed';
        setAuthError(message);
        setIsLoading(false);
        throw err;
      }
    },
    [logout, restoreSession]
  );

  /**
   * Register a new user and organization, then restore session.
   */
  const register = useCallback(
    async (data: RegisterRequest): Promise<void> => {
      setIsLoading(true);
      setAuthError(null);
      try {
        const tokenResponse = await registerUser(data);
        setAuthToken(tokenResponse.access_token);
        await restoreSession();
      } catch (err: unknown) {
        logout();
        const message = err instanceof Error ? err.message : 'Registration failed';
        setAuthError(message);
        setIsLoading(false);
        throw err;
      }
    },
    [logout, restoreSession]
  );

  /**
   * Switch the active organization to another membership belonging to the user.
   */
  const switchOrganization = useCallback(
    (orgId: string): void => {
      if (!isAuthenticated || !user) {
        throw new Error('Cannot switch organization: user is not authenticated.');
      }

      const targetOrg = organizations.find((org) => org.organization_id === orgId);
      if (!targetOrg) {
        throw new Error(`Unauthorized organization selection: user does not hold membership in organization '${orgId}'.`);
      }

      setSelectedOrgId(targetOrg.organization_id);
      setCurrentOrg(targetOrg);
    },
    [isAuthenticated, user, organizations]
  );

  // Register unauthorized (401) listener on mount
  useEffect(() => {
    setUnauthorizedHandler(() => {
      // Direct local session and state cleanup on 401 without re-invoking network logout
      clearAuthSession();
      setUser(null);
      setOrganizations([]);
      setCurrentOrg(null);
      setIsAuthenticated(false);
    });

    return () => {
      setUnauthorizedHandler(null);
    };
  }, []);

  // Initial session restoration on startup
  useEffect(() => {
    let isMounted = true;
    const init = async () => {
      await Promise.resolve();
      if (isMounted) {
        await restoreSession();
      }
    };
    init();
    return () => {
      isMounted = false;
    };
  }, [restoreSession]);

  const value: AuthContextType = {
    user,
    organizations,
    currentOrg,
    isAuthenticated,
    isLoading,
    authError,
    login,
    register,
    logout,
    switchOrganization,
    restoreSession,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

/**
 * Hook to consume the authentication context.
 */
export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}


