import React from 'react';
import './LandingPage.css';
import {
  Sparkles,
  Mic,
  FileText,
  CheckSquare,
  Search,
  Layers,
  Clock,
  Shield,
  ExternalLink,
  ArrowRight,
  Upload,
  Cpu,
  MessageSquare,
  CheckCircle2,
} from 'lucide-react';

interface LandingPageProps {
  onSignIn: () => void;
  onGetStarted: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onSignIn, onGetStarted }) => {
  return (
    <div className="landing-page">
      {/* Ambient Background Layers */}
      <div className="lp-bg-glow" aria-hidden="true" />
      <div className="lp-bg-grid" aria-hidden="true" />

      {/* Navigation Bar */}
      <header className="navbar-wrapper">
        <nav className="navbar" aria-label="Primary">
          <a href="#home" className="logo">
            <Sparkles size={28} /> <span>MEMORA</span>
          </a>
          <ul className="nav-links">
            <li><a href="#">Home</a></li>
            <li><a href="#features">Features</a></li>
            <li><a href="#how-it-works">How It Works</a></li>
            <li><a href="#about">About</a></li>
          </ul>
          <div className="nav-actions">
            <button type="button" className="btn btn-outline" onClick={onSignIn}>
              Sign In
            </button>
            <button type="button" className="btn btn-primary" onClick={onGetStarted}>
              Get Started
            </button>
          </div>
        </nav>
      </header>

      <main className="lp-main">
        {/* Hero Section */}
        <section className="hero" id="home">
          <div className="hero-content">
            <h1>Turn Every Meeting Into Lasting Intelligence.</h1>
            <p>
              MEMORA transforms meetings into searchable organizational knowledge through multilingual transcription, AI-powered summaries, decision extraction, and intelligent question answering.
            </p>
            <div className="hero-buttons">
              <button type="button" className="btn btn-primary btn-lg" onClick={onGetStarted}>
                <span>Get Started</span>
                <ArrowRight size={16} />
              </button>
              <button
                type="button"
                className="btn btn-outline btn-lg"
                onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              >
                Explore Features
              </button>
            </div>
          </div>

          <div className="hero-illustration" aria-hidden="true">
            <div className="illustration-mockup">
              <div className="mockup-topbar">
                <div className="mockup-dots">
                  <span />
                  <span />
                  <span />
                </div>
                <div className="mockup-title">
                  <Sparkles size={13} />
                  <span>MEMORA &middot; Meeting Intelligence Preview</span>
                </div>
                <div className="mockup-pill">
                  <Shield size={11} />
                  <span>PII Redacted</span>
                </div>
              </div>

              <div className="mockup-body">
                <div className="mockup-pane mockup-transcript">
                  <div className="mockup-pane-header">
                    <Mic size={13} />
                    <span>Multilingual Transcript</span>
                    <span className="mockup-lang-badge">EN / HI / GU</span>
                  </div>
                  <div className="mockup-utterance">
                    <span className="mockup-time">02:14</span>
                    <p>Let&apos;s finalize the architecture rollout by Friday and share updates with the team.</p>
                  </div>
                  <div className="mockup-utterance highlighted">
                    <span className="mockup-time">04:18</span>
                    <p>
                      Agreed &mdash; hum deployment schedule lock kar dete hain aur{' '}
                      <span className="mockup-redacted">[EMAIL_REDACTED]</span> par confirmation bhej denge.
                    </p>
                  </div>
                  <div className="mockup-utterance">
                    <span className="mockup-time">05:02</span>
                    <p>Action item assigned with deadline logged to organizational memory.</p>
                  </div>
                </div>

                <div className="mockup-pane mockup-intelligence">
                  <div className="mockup-pane-header">
                    <Sparkles size={13} />
                    <span>Structured Intelligence</span>
                  </div>

                  <div className="mockup-intel-block">
                    <div className="mockup-intel-label">
                      <FileText size={12} />
                      <span>Executive Summary</span>
                    </div>
                    <p>
                      Team aligned on Friday architecture rollout and confirmed automated confirmation workflow with privacy-safe transcript grounding.
                    </p>
                  </div>

                  <div className="mockup-intel-block">
                    <div className="mockup-intel-label">
                      <CheckCircle2 size={12} />
                      <span>Extracted Decision</span>
                    </div>
                    <p>Lock deployment schedule for Friday release.</p>
                    <div className="mockup-citation">
                      <Clock size={11} />
                      <span>Evidence &middot; 04:18</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="features" id="features">
          <div className="lp-section-inner">
            <div className="lp-section-header">
              <h2>Features</h2>
            </div>
            <div className="feature-grid">
              <div className="feature-card">
                <div className="feature-icon" aria-hidden="true">
                  <Mic size={18} />
                </div>
                <h3>Multilingual Meeting Transcription</h3>
                <p>Accurate speech‑to‑text in English, Hindi, Gujarati and Hinglish.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon" aria-hidden="true">
                  <FileText size={18} />
                </div>
                <h3>AI‑Powered Meeting Summaries</h3>
                <p>Executive‑level summaries generated by Gemini.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon" aria-hidden="true">
                  <CheckSquare size={18} />
                </div>
                <h3>Automatic Decision &amp; Action Extraction</h3>
                <p>Identify decisions, owners, deadlines and action items.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon" aria-hidden="true">
                  <Search size={18} />
                </div>
                <h3>Intelligent Meeting Search</h3>
                <p>Semantic search over all meetings with timestamp citations.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon" aria-hidden="true">
                  <Layers size={18} />
                </div>
                <h3>Organizational Memory Across Meetings</h3>
                <p>Cross‑meeting context lets you reuse knowledge.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon" aria-hidden="true">
                  <Clock size={18} />
                </div>
                <h3>Evidence‑Based Answers with Timestamps</h3>
                <p>Answers are grounded in the original transcript.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon" aria-hidden="true">
                  <Shield size={18} />
                </div>
                <h3>Privacy‑Aware Meeting Processing</h3>
                <p>PII redaction pipeline protects sensitive data.</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon" aria-hidden="true">
                  <ExternalLink size={18} />
                </div>
                <h3>External Knowledge Intelligence</h3>
                <p>Leverage Gemini to augment answers with web knowledge.</p>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works */}
        <section className="how-it-works" id="how-it-works">
          <div className="lp-section-inner">
            <div className="lp-section-header">
              <h2>How It Works</h2>
            </div>
            <div className="steps">
              <div className="step">
                <div className="step-header">
                  <div className="step-number">1</div>
                  <div className="step-icon" aria-hidden="true">
                    <Upload size={17} />
                  </div>
                </div>
                <h3>Upload your meeting recording</h3>
                <p>Supported audio/video formats are ingested securely.</p>
              </div>
              <div className="step">
                <div className="step-header">
                  <div className="step-number">2</div>
                  <div className="step-icon" aria-hidden="true">
                    <Cpu size={17} />
                  </div>
                </div>
                <h3>Generate structured intelligence</h3>
                <p>Transcription, cleaning, Gemini extraction and storage happen automatically.</p>
              </div>
              <div className="step">
                <div className="step-header">
                  <div className="step-number">3</div>
                  <div className="step-icon" aria-hidden="true">
                    <MessageSquare size={17} />
                  </div>
                </div>
                <h3>Ask questions and discover insights</h3>
                <p>Search or query the knowledge base to retrieve precise answers.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Why MEMORA */}
        <section className="why-memora" id="about">
          <div className="lp-section-inner">
            <div className="why-memora-card">
              <h2>Why MEMORA?</h2>
              <p>
                Traditional transcription tools give you raw text. MEMORA goes beyond by turning those transcripts into an <strong>organizational memory</strong> – structured decisions, actions, and searchable insights that are linked to the original evidence. This turns meetings into a living knowledge base, not just a file archive.
              </p>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="cta">
          <div className="lp-section-inner">
            <div className="cta-card">
              <h2>Your Meetings Deserve More Than Just Transcripts.</h2>
              <div className="cta-actions">
                <button type="button" className="btn btn-primary btn-lg" onClick={onGetStarted}>
                  <span>Get Started</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-top">
            <div className="footer-brand">
              <Sparkles size={24} /> <span>MEMORA</span>
            </div>
            <ul className="footer-links">
              <li><a href="#home">Home</a></li>
              <li><a href="#features">Features</a></li>
              <li><a href="#how-it-works">How It Works</a></li>
              <li><a href="#about">About</a></li>
            </ul>
          </div>
          <div className="footer-bottom">
            <p>© {new Date().getFullYear()} MEMORA – Multilingual Meeting Intelligence &amp; Organizational Memory System.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
