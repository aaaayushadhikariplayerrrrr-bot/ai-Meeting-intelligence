import React from 'react';
import { Building, ArrowRight, Shield, LogOut, AlertTriangle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './AuthScreen.css';

export const OrgSelectionScreen: React.FC = () => {
  const { user, organizations, switchOrganization, logout } = useAuth();

  const handleSelect = (orgId: string) => {
    try {
      switchOrganization(orgId);
    } catch (err: unknown) {
      alert(err instanceof Error ? err.message : 'Failed to switch organization');
    }
  };

  if (!organizations || organizations.length === 0) {
    return (
      <div className="auth-page-wrapper">
        <div className="org-selection-card">
          <div className="auth-header">
            <div className="auth-brand-icon" style={{ background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444' }}>
              <AlertTriangle size={24} />
            </div>
            <h1 className="auth-title">No Active Organization</h1>
            <p className="auth-subtitle">
              Account <strong>{user?.email}</strong> is verified, but holds no active organization memberships.
            </p>
          </div>

          <div className="auth-error-banner">
            <AlertTriangle size={16} />
            <span>
              Tenant membership is required to access meetings and media. Please contact an organization administrator to be added to a workspace.
            </span>
          </div>

          <button
            type="button"
            className="auth-submit-btn"
            style={{ background: '#1e293b', border: '1px solid var(--border-subtle)' }}
            onClick={logout}
          >
            <LogOut size={16} />
            <span>Sign Out</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-page-wrapper">
      <div className="org-selection-card">
        <div className="auth-header">
          <div className="auth-brand-icon">
            <Building size={24} />
          </div>
          <h1 className="auth-title">Select Active Workspace</h1>
          <p className="auth-subtitle">
            Signed in as <strong>{user?.full_name}</strong> ({user?.email})
          </p>
        </div>

        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textAlign: 'center', lineHeight: 1.5 }}>
          Your account belongs to multiple organizations. Please choose a workspace to enter the meeting intelligence dashboard:
        </p>

        <div className="org-list">
          {organizations.map((org) => {
            const isAdmin = org.role === 'ORG_ADMIN';
            return (
              <button
                key={org.organization_id}
                type="button"
                className="org-card-btn"
                onClick={() => handleSelect(org.organization_id)}
              >
                <div className="org-card-info">
                  <div className="org-card-icon">
                    <Building size={18} />
                  </div>
                  <div>
                    <div className="org-card-title">{org.organization_name}</div>
                    <div className="org-card-slug">/{org.organization_slug}</div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span className={`org-card-badge ${isAdmin ? 'admin' : ''}`}>
                    <Shield size={10} style={{ display: 'inline', marginRight: '3px' }} />
                    {org.role}
                  </span>
                  <ArrowRight size={16} color="#6366f1" />
                </div>
              </button>
            );
          })}
        </div>

        <div className="auth-footer" style={{ marginTop: '0.5rem' }}>
          <button
            type="button"
            className="auth-switch-link"
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: '#94a3b8' }}
            onClick={logout}
          >
            <LogOut size={14} />
            <span>Sign in with a different account</span>
          </button>
        </div>
      </div>
    </div>
  );
};
