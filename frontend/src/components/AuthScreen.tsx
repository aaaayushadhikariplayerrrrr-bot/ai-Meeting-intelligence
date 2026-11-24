import React, { useState } from 'react';
import {
  Sparkles,
  Mail,
  Lock,
  User,
  Building,
  Users,
  Eye,
  EyeOff,
  AlertTriangle,
  ArrowRight,
  RefreshCw,
  Shield,
  LogIn,
  UserPlus,
  CheckCircle2,
  Globe,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { lookupDomain } from '../services/api';
import type { RegistrationType, DomainLookupResponse } from '../types';
import './AuthScreen.css';

export interface AuthScreenProps {
  defaultMode?: 'login' | 'register';
}

export const AuthScreen: React.FC<AuthScreenProps> = ({ defaultMode = 'login' }) => {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<'login' | 'register'>(defaultMode);

  // Registration account pathway: CREATE_ORG | JOIN_ORG | INDIVIDUAL
  const [regType, setRegType] = useState<RegistrationType>('CREATE_ORG');

  // Form input states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [orgName, setOrgName] = useState('');
  const [companyDomain, setCompanyDomain] = useState('');

  // Domain lookup feedback
  const [domainStatus, setDomainStatus] = useState<DomainLookupResponse | null>(null);
  const [isCheckingDomain, setIsCheckingDomain] = useState(false);

  // UI behavior states
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const handleSwitchMode = (targetMode: 'login' | 'register') => {
    setMode(targetMode);
    setErrorMessage(null);
    setSuccessMessage(null);
    setPassword('');
  };

  // Instant domain lookup when joining an organization
  const handleDomainBlur = async () => {
    if (regType !== 'JOIN_ORG' || !companyDomain.trim()) return;
    setIsCheckingDomain(true);
    try {
      const res = await lookupDomain(companyDomain.trim());
      setDomainStatus(res);
      if (!res.exists) {
        setErrorMessage(`No verified organization found registered with domain '${res.domain}'. Please ask your administrator to create the organization first.`);
      } else if (!res.is_verified) {
        setErrorMessage(`The organization for '${res.domain}' is not yet verified. An administrator must verify domain ownership first.`);
      } else {
        setErrorMessage(null);
      }
    } catch {
      // Ignore lookup failure
    } finally {
      setIsCheckingDomain(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    setErrorMessage(null);
    setSuccessMessage(null);

    const trimmedEmail = email.trim().toLowerCase();
    if (!trimmedEmail) {
      setErrorMessage('Please enter a valid email address.');
      return;
    }

    if (!password) {
      setErrorMessage('Please enter your password.');
      return;
    }

    if (mode === 'register') {
      const trimmedName = fullName.trim();
      if (!trimmedName) {
        setErrorMessage('Please enter your full name.');
        return;
      }
      if (password.length < 8) {
        setErrorMessage('Password must be at least 8 characters long.');
        return;
      }
      if (password.length > 128) {
        setErrorMessage('Password cannot exceed 128 characters.');
        return;
      }

      // 1. Create Organization pathway
      if (regType === 'CREATE_ORG') {
        const trimmedOrg = orgName.trim();
        const trimmedDomain = companyDomain.trim().toLowerCase().replace(/^@/, '').replace(/^https?:\/\//, '').replace(/^www\./, '').split('/')[0];

        if (!trimmedOrg) {
          setErrorMessage('Please enter an organization or company name.');
          return;
        }

        if (trimmedDomain) {
          // Check public domain
          const isPublic = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com'].includes(trimmedDomain);
          if (isPublic) {
            setErrorMessage('Personal email domains (such as Gmail or Yahoo) cannot be used as company domains. Please register an Individual Workspace instead.');
            return;
          }
          const emailDomain = trimmedEmail.split('@')[1];
          if (emailDomain !== trimmedDomain) {
            setErrorMessage(`Your email domain (@${emailDomain}) must match the company domain (${trimmedDomain}).`);
            return;
          }
        }

        setIsSubmitting(true);
        try {
          await register({
            registration_type: 'CREATE_ORG',
            email: trimmedEmail,
            full_name: trimmedName,
            password,
            organization_name: trimmedOrg,
            company_domain: trimmedDomain || undefined,
          });
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : 'Registration failed. Please try again.';
          setErrorMessage(msg);
        } finally {
          setIsSubmitting(false);
        }
      }

      // 2. Join Organization pathway
      else if (regType === 'JOIN_ORG') {
        const trimmedDomain = companyDomain.trim().toLowerCase().replace(/^@/, '').replace(/^https?:\/\//, '').replace(/^www\./, '').split('/')[0];
        if (!trimmedDomain) {
          setErrorMessage('Please enter your company email domain to find your organization.');
          return;
        }

        const isPublic = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com'].includes(trimmedDomain);
        if (isPublic) {
          setErrorMessage('Personal email providers (e.g. Gmail) cannot join corporate organizations. Please use your company email or choose Individual Workspace.');
          return;
        }

        const emailDomain = trimmedEmail.split('@')[1];
        if (emailDomain !== trimmedDomain) {
          setErrorMessage(`Your email domain (@${emailDomain}) does not match the entered company domain (${trimmedDomain}).`);
          return;
        }

        setIsSubmitting(true);
        try {
          await register({
            registration_type: 'JOIN_ORG',
            email: trimmedEmail,
            full_name: trimmedName,
            password,
            company_domain: trimmedDomain,
          });
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : 'Failed to join organization. Please verify the domain.';
          setErrorMessage(msg);
        } finally {
          setIsSubmitting(false);
        }
      }

      // 3. Individual Workspace pathway
      else if (regType === 'INDIVIDUAL') {
        setIsSubmitting(true);
        try {
          await register({
            registration_type: 'INDIVIDUAL',
            email: trimmedEmail,
            full_name: trimmedName,
            password,
          });
        } catch (err: unknown) {
          const msg = err instanceof Error ? err.message : 'Registration failed. Please try again.';
          setErrorMessage(msg);
        } finally {
          setIsSubmitting(false);
        }
      }
    } else {
      // Login mode - simple, asks only for email and password
      setIsSubmitting(true);
      try {
        await login({
          email: trimmedEmail,
          password,
        });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Sign in failed. Please verify your credentials.';
        setErrorMessage(msg);
      } finally {
        setIsSubmitting(false);
      }
    }
  };

  return (
    <div className="auth-page-wrapper">
      <div className="auth-card">
        {/* Brand Header */}
        <div className="auth-header">
          <div className="auth-brand-icon">
            <Sparkles size={24} />
          </div>
          <h1 className="auth-title">MEMORA</h1>
          <p className="auth-subtitle">
            Organizational Memory & Dual-Source Multilingual RAG Platform
          </p>
        </div>

        {/* Sign In vs Create Account Tabs */}
        <div className="auth-mode-tabs" role="tablist" aria-label="Authentication Mode">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'login'}
            className={`auth-tab-btn ${mode === 'login' ? 'active' : ''}`}
            onClick={() => handleSwitchMode('login')}
            disabled={isSubmitting}
          >
            <LogIn size={15} />
            <span>Sign In</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'register'}
            className={`auth-tab-btn ${mode === 'register' ? 'active' : ''}`}
            onClick={() => handleSwitchMode('register')}
            disabled={isSubmitting}
          >
            <UserPlus size={15} />
            <span>Create Account</span>
          </button>
        </div>

        {/* Account Pathway Selection in Register Mode */}
        {mode === 'register' && (
          <div className="auth-pathway-tabs" role="tablist" aria-label="Workspace Type">
            <button
              type="button"
              className={`auth-pathway-btn ${regType === 'CREATE_ORG' ? 'active' : ''}`}
              onClick={() => { setRegType('CREATE_ORG'); setErrorMessage(null); }}
              title="Create a new verified company workspace as an administrator"
            >
              <Building size={13} />
              <span>Create Org</span>
            </button>
            <button
              type="button"
              className={`auth-pathway-btn ${regType === 'JOIN_ORG' ? 'active' : ''}`}
              onClick={() => { setRegType('JOIN_ORG'); setErrorMessage(null); }}
              title="Join an existing verified company workspace with your work email"
            >
              <Users size={13} />
              <span>Join Org</span>
            </button>
            <button
              type="button"
              className={`auth-pathway-btn ${regType === 'INDIVIDUAL' ? 'active' : ''}`}
              onClick={() => { setRegType('INDIVIDUAL'); setErrorMessage(null); }}
              title="Individual personal workspace without organization affiliation"
            >
              <User size={13} />
              <span>Individual</span>
            </button>
          </div>
        )}

        {/* Error Alert Banner */}
        {errorMessage && (
          <div className="auth-error-banner" role="alert">
            <AlertTriangle size={16} />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Success Alert Banner */}
        {successMessage && (
          <div className="auth-success-banner" role="status">
            <CheckCircle2 size={16} color="#34d399" />
            <span>{successMessage}</span>
          </div>
        )}

        {/* Authentication Form */}
        <form onSubmit={handleSubmit} className="auth-form" noValidate>
          {mode === 'register' && (
            <div className="auth-field-group">
              <label className="auth-label" htmlFor="auth-fullname">
                <User size={14} /> Full Name
              </label>
              <div className="auth-input-wrapper">
                <input
                  id="auth-fullname"
                  type="text"
                  className="auth-input"
                  placeholder="Jane Doe"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  disabled={isSubmitting}
                  autoComplete="name"
                  required
                />
              </div>
            </div>
          )}

          {/* Create Organization Specific Fields */}
          {mode === 'register' && regType === 'CREATE_ORG' && (
            <>
              <div className="auth-field-group">
                <label className="auth-label" htmlFor="auth-orgname">
                  <Building size={14} /> Organization / Company Name
                </label>
                <div className="auth-input-wrapper">
                  <input
                    id="auth-orgname"
                    type="text"
                    className="auth-input"
                    placeholder="e.g. Acme Technologies"
                    value={orgName}
                    onChange={(e) => setOrgName(e.target.value)}
                    disabled={isSubmitting}
                    autoComplete="organization"
                    required
                  />
                </div>
              </div>

              <div className="auth-field-group">
                <label className="auth-label" htmlFor="auth-companydomain">
                  <Globe size={14} /> Company Domain (e.g. acmetech.com)
                </label>
                <div className="auth-input-wrapper">
                  <input
                    id="auth-companydomain"
                    type="text"
                    className="auth-input"
                    placeholder="acmetech.com"
                    value={companyDomain}
                    onChange={(e) => setCompanyDomain(e.target.value)}
                    disabled={isSubmitting}
                    autoComplete="url"
                  />
                </div>
                <span className="auth-field-hint">
                  The initial creator of this verified domain receives the Admin role.
                </span>
              </div>
            </>
          )}

          {/* Join Organization Specific Fields */}
          {mode === 'register' && regType === 'JOIN_ORG' && (
            <div className="auth-field-group">
              <label className="auth-label" htmlFor="auth-joindomain">
                <Globe size={14} /> Company Domain to Join
              </label>
              <div className="auth-input-wrapper">
                <input
                  id="auth-joindomain"
                  type="text"
                  className="auth-input"
                  placeholder="company.com"
                  value={companyDomain}
                  onChange={(e) => setCompanyDomain(e.target.value)}
                  onBlur={handleDomainBlur}
                  disabled={isSubmitting}
                  autoComplete="url"
                  required
                />
                {isCheckingDomain && <RefreshCw size={14} className="spinning auth-input-icon" />}
              </div>
              {domainStatus && domainStatus.exists && domainStatus.is_verified && (
                <span className="auth-field-hint success" style={{ color: '#34d399' }}>
                  <CheckCircle2 size={12} style={{ display: 'inline', marginRight: 4 }} />
                  Verified organization: {domainStatus.organization_name}
                </span>
              )}
              <span className="auth-field-hint">
                Employees join with Member permissions (never automatic Admin).
              </span>
            </div>
          )}

          {/* Email Address */}
          <div className="auth-field-group">
            <label className="auth-label" htmlFor="auth-email">
              <Mail size={14} />{' '}
              {mode === 'register' && (regType === 'CREATE_ORG' || regType === 'JOIN_ORG')
                ? 'Company Email Address'
                : 'Email Address'}
            </label>
            <div className="auth-input-wrapper">
              <input
                id="auth-email"
                type="email"
                className="auth-input"
                placeholder={
                  mode === 'register' && (regType === 'CREATE_ORG' || regType === 'JOIN_ORG')
                    ? 'name@company.com'
                    : 'name@example.com'
                }
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isSubmitting}
                autoComplete="email"
                required
              />
            </div>
          </div>

          {/* Password */}
          <div className="auth-field-group">
            <label className="auth-label" htmlFor="auth-password">
              <Lock size={14} /> Password
            </label>
            <div className="auth-input-wrapper">
              <input
                id="auth-password"
                type={showPassword ? 'text' : 'password'}
                className="auth-input has-toggle"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isSubmitting}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                required
              />
              <button
                type="button"
                className="auth-password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                title={showPassword ? 'Hide password' : 'Show password'}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                tabIndex={-1}
              >
                {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {mode === 'register' && (
              <span className="auth-field-hint">Must be 8 to 128 characters</span>
            )}
          </div>

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <RefreshCw size={16} className="spinning" />
                <span>{mode === 'login' ? 'Signing in...' : 'Processing...'}</span>
              </>
            ) : (
              <>
                <span>
                  {mode === 'login'
                    ? 'Sign In to Workspace'
                    : regType === 'CREATE_ORG'
                    ? 'Create & Verify Organization'
                    : regType === 'JOIN_ORG'
                    ? 'Join as Team Member'
                    : 'Launch Individual Workspace'}
                </span>
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>

        {/* Switch Link & Security Note */}
        <div className="auth-footer">
          {mode === 'login' ? (
            <p>
              Don't have an account yet?
              <button
                type="button"
                className="auth-switch-link"
                onClick={() => handleSwitchMode('register')}
                disabled={isSubmitting}
              >
                Create one
              </button>
            </p>
          ) : (
            <p>
              Already have an account?
              <button
                type="button"
                className="auth-switch-link"
                onClick={() => handleSwitchMode('login')}
                disabled={isSubmitting}
              >
                Sign in
              </button>
            </p>
          )}

          <div className="auth-security-note">
            <Shield size={12} />
            <span>
              {regType === 'INDIVIDUAL' && mode === 'register'
                ? 'Independent Personal Workspace • Zero Data Leakage'
                : 'Verified Corporate Domains • Strict Tenant Isolation • RBAC'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
