import React, { useState } from 'react';
import {
  Sparkles,
  Mail,
  Lock,
  User,
  Building,
  Eye,
  EyeOff,
  AlertTriangle,
  ArrowRight,
  RefreshCw,
  Shield,
  LogIn,
  UserPlus,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './AuthScreen.css';

export interface AuthScreenProps {
  defaultMode?: 'login' | 'register';
}

export const AuthScreen: React.FC<AuthScreenProps> = ({ defaultMode = 'login' }) => {
  const { login, register } = useAuth();
  const [mode, setMode] = useState<'login' | 'register'>(defaultMode);

  // Form input states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [orgName, setOrgName] = useState('');

  // UI behavior states
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSwitchMode = (targetMode: 'login' | 'register') => {
    setMode(targetMode);
    setErrorMessage(null);
    setPassword('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;

    setErrorMessage(null);

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
      const trimmedOrg = orgName.trim();

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
      if (!trimmedOrg) {
        setErrorMessage('Please enter an organization or workspace name.');
        return;
      }

      setIsSubmitting(true);
      try {
        await register({
          email: trimmedEmail,
          full_name: trimmedName,
          password,
          organization_name: trimmedOrg,
        });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Registration failed. Please try again.';
        setErrorMessage(msg);
      } finally {
        setIsSubmitting(false);
      }
    } else {
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
          <h1 className="auth-title">Meeting Intelligence</h1>
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

        {/* Error Alert Banner */}
        {errorMessage && (
          <div className="auth-error-banner" role="alert">
            <AlertTriangle size={16} />
            <span>{errorMessage}</span>
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

          <div className="auth-field-group">
            <label className="auth-label" htmlFor="auth-email">
              <Mail size={14} /> Email Address
            </label>
            <div className="auth-input-wrapper">
              <input
                id="auth-email"
                type="email"
                className="auth-input"
                placeholder="name@company.internal"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isSubmitting}
                autoComplete="email"
                required
              />
            </div>
          </div>

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

          {mode === 'register' && (
            <div className="auth-field-group">
              <label className="auth-label" htmlFor="auth-orgname">
                <Building size={14} /> Organization / Workspace Name
              </label>
              <div className="auth-input-wrapper">
                <input
                  id="auth-orgname"
                  type="text"
                  className="auth-input"
                  placeholder="e.g. Acme Innovations"
                  value={orgName}
                  onChange={(e) => setOrgName(e.target.value)}
                  disabled={isSubmitting}
                  autoComplete="organization"
                  required
                />
              </div>
            </div>
          )}

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <RefreshCw size={16} className="spinning" />
                <span>{mode === 'login' ? 'Signing in...' : 'Creating account...'}</span>
              </>
            ) : (
              <>
                <span>{mode === 'login' ? 'Sign In to Workspace' : 'Create Account & Organization'}</span>
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
            <span>Argon2id password hashing &bull; Strict Tenant Isolation</span>
          </div>
        </div>
      </div>
    </div>
  );
};
