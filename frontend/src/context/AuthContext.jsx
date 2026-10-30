/**
 * Authentication Context for EchoSafe.
 */

import React, { createContext, useContext, useState, useEffect } from "react";
import { api, getStoredToken, setStoredToken } from "../api";

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(getStoredToken());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function initAuth() {
      const stored = getStoredToken();
      if (stored) {
        try {
          const userData = await api.getMe();
          setUser(userData);
          setLoading(false);
          return;
        } catch (err) {
          console.warn("Stored token expired, auto-logging in demo account:", err.message);
          setStoredToken(null);
        }
      }

      // Default auto-login for frictionless demo experience: user_positive
      try {
        const demoAuth = await api.demoLogin("positive");
        setStoredToken(demoAuth.access_token);
        setToken(demoAuth.access_token);
        const u = await api.getMe();
        setUser(u);
      } catch (e) {
        console.error("Auto demo-login failed:", e);
      } finally {
        setLoading(false);
      }
    }

    initAuth();
  }, []);

  async function login(username, password) {
    const res = await api.login(username, password);
    setStoredToken(res.access_token);
    setToken(res.access_token);
    const userData = await api.getMe();
    setUser(userData);
    return userData;
  }

  async function switchDemoRole(roleKey) {
    setLoading(true);
    try {
      const res = await api.demoLogin(roleKey);
      setStoredToken(res.access_token);
      setToken(res.access_token);
      const userData = await api.getMe();
      setUser(userData);
    } finally {
      setLoading(false);
    }
  }

  function logout() {
    setStoredToken(null);
    setToken(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, login, logout, switchDemoRole }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
