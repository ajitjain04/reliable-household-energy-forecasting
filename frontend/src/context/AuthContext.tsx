import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { api } from '../api/client';
import { UserProfile } from '../api/types';

interface AuthContextType {
  user: UserProfile | null;
  token: string | null;
  loading: boolean;
  isAuthenticated: boolean;
  isDeveloper: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  isAuthModalOpen: boolean;
  authModalDestination: string | null;
  openAuthModal: (intendedDestination?: string) => void;
  closeAuthModal: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [token, setToken] = useState<string | null>(api.getToken());
  const [loading, setLoading] = useState<boolean>(true);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState<boolean>(false);
  const [authModalDestination, setAuthModalDestination] = useState<string | null>(null);

  const openAuthModal = useCallback((intendedDestination?: string) => {
    setAuthModalDestination(intendedDestination || '/forecast/new');
    setIsAuthModalOpen(true);
  }, []);

  const closeAuthModal = useCallback(() => {
    setIsAuthModalOpen(false);
    setAuthModalDestination(null);
  }, []);

  const refreshUser = useCallback(async () => {
    const currentToken = api.getToken();
    if (!currentToken) {
      setUser(null);
      setToken(null);
      setLoading(false);
      return;
    }

    try {
      const profile = await api.getMe();
      setUser(profile);
      setToken(currentToken);
    } catch {
      // Invalid or expired token: clear session
      api.setToken(null);
      setUser(null);
      setToken(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = async (username: string, password: string) => {
    const response = await api.login(username, password);
    setToken(response.access_token);
    setUser(response.user);
    setIsAuthModalOpen(false);
    setAuthModalDestination(null);
  };

  const register = async (username: string, password: string) => {
    await api.register(username, password);
    // Automatically log in following registration
    await login(username, password);
  };

  const logout = async () => {
    await api.logout();
    setToken(null);
    setUser(null);
    setIsAuthModalOpen(false);
    setAuthModalDestination(null);
  };

  const isAuthenticated = Boolean(user && token);
  const isDeveloper = Boolean(user && user.role === 'developer');

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        isAuthenticated,
        isDeveloper,
        login,
        register,
        logout,
        refreshUser,
        isAuthModalOpen,
        authModalDestination,
        openAuthModal,
        closeAuthModal,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
