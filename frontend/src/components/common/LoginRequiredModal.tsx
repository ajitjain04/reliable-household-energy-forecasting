import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, LogIn, UserPlus, X, Zap } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

interface LoginRequiredModalProps {
  isOpen?: boolean;
  intendedDestination?: string | null;
  onClose?: () => void;
}

export const LoginRequiredModal: React.FC<LoginRequiredModalProps> = ({
  isOpen: propIsOpen,
  intendedDestination: propDestination,
  onClose: propOnClose,
}) => {
  const { isAuthModalOpen, authModalDestination, closeAuthModal } = useAuth();
  const navigate = useNavigate();

  // Support both context-driven state and component-prop-driven state
  const isOpen = propIsOpen !== undefined ? propIsOpen : isAuthModalOpen;
  const destination = propDestination !== undefined ? propDestination : (authModalDestination || '/forecast/new');

  // Handle Cancel action: return visitor to the page they were viewing
  const handleCancel = () => {
    if (propOnClose) {
      propOnClose();
    } else {
      closeAuthModal();
    }
  };

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleCancel();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSignIn = () => {
    closeAuthModal();
    if (propOnClose) propOnClose();
    navigate('/login', {
      state: { from: { pathname: destination } },
    });
  };

  const handleCreateAccount = () => {
    closeAuthModal();
    if (propOnClose) propOnClose();
    navigate('/register', {
      state: {
        from: { pathname: destination },
        mode: 'register',
      },
    });
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="auth-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs animate-fadeIn"
      onClick={handleCancel}
    >
      <div
        className="relative w-full max-w-md bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-800 overflow-hidden transform transition-all animate-scaleUp"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Subtle decorative top border accent */}
        <div className="h-1.5 w-full bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600" />

        {/* Close / Cancel corner button */}
        <button
          type="button"
          onClick={handleCancel}
          aria-label="Close modal"
          className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="p-7">
          {/* Header Icon & Title */}
          <div className="flex items-start gap-4 mb-4">
            <div className="w-12 h-12 rounded-xl bg-blue-50 dark:bg-blue-950/50 border border-blue-200 dark:border-blue-800 flex items-center justify-center text-blue-600 dark:text-blue-400 flex-shrink-0">
              <Lock className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-mono font-semibold uppercase bg-blue-100/70 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 mb-1 border border-blue-200 dark:border-blue-800">
                <Zap className="w-3 h-3 text-blue-600 dark:text-blue-400" />
                Protected Household Action
              </div>
              <h2
                id="auth-modal-title"
                className="text-lg font-bold text-slate-900 dark:text-white tracking-tight"
              >
                Sign in required
              </h2>
            </div>
          </div>

          {/* Explanation Body */}
          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed mb-6">
            Please sign in or create a household account to generate forecasts and access your saved reports.
          </p>

          {/* Action Buttons */}
          <div className="space-y-2.5">
            <button
              type="button"
              onClick={handleSignIn}
              className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-xs hover:shadow transition-all"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Sign In</span>
            </button>

            <button
              type="button"
              onClick={handleCreateAccount}
              className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 font-semibold text-xs border border-slate-300 dark:border-slate-700 shadow-2xs hover:border-slate-400 dark:hover:border-slate-600 transition-all"
            >
              <UserPlus className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
              <span>Create Account</span>
            </button>

            <button
              type="button"
              onClick={handleCancel}
              className="w-full inline-flex items-center justify-center px-4 py-2 rounded-xl text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-slate-100/80 dark:hover:bg-slate-800/80 font-medium text-xs transition-colors"
            >
              Cancel
            </button>
          </div>

          {/* Destination Preservation Note */}
          {destination && destination !== '/' && (
            <div className="mt-5 pt-4 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400 flex items-center justify-center gap-1.5 font-mono">
              <span className="text-slate-400 dark:text-slate-500">Intended Destination:</span>
              <span className="font-semibold text-slate-700 dark:text-slate-200 truncate max-w-[200px]">
                {destination}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LoginRequiredModal;
