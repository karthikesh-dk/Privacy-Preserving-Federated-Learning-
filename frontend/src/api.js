/**
 * Centralized API Client for EchoSafe.
 */

const API_BASE = "";

export function getStoredToken() {
  return localStorage.getItem("echosafe_token");
}

export function setStoredToken(token) {
  if (token) {
    localStorage.setItem("echosafe_token", token);
  } else {
    localStorage.removeItem("echosafe_token");
  }
}

async function request(endpoint, options = {}) {
  const token = getStoredToken();
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {})
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers
  });

  if (response.status === 401) {
    // Token expired or invalid
    setStoredToken(null);
  }

  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const errorMsg = data?.detail || `HTTP Error ${response.status}`;
    throw new Error(errorMsg);
  }
  return data;
}

export const api = {
  // Auth
  login: (username, password) => request("/api/auth/login", { method: "POST", body: JSON.stringify({ username, password }) }),
  register: (payload) => request("/api/auth/register", { method: "POST", body: JSON.stringify(payload) }),
  getMe: () => request("/api/auth/me"),
  demoLogin: (role) => request(`/api/auth/demo-login/${role}`, { method: "POST" }),

  // Posts & Feed
  getFeed: (postType) => request(`/api/posts${postType ? `?post_type=${postType}` : ""}`),
  createPost: (payload) => request("/api/posts", { method: "POST", body: JSON.stringify(payload) }),
  toggleLike: (postId) => request(`/api/posts/${postId}/like`, { method: "POST" }),

  // Comments & Pipeline
  getComments: (postId) => request(`/api/posts/${postId}/comments`),
  postComment: (payload) => request("/api/comments", { method: "POST", body: JSON.stringify(payload) }),
  submitAppeal: (commentId, reason) => request(`/api/comments/${commentId}/appeal`, { method: "POST", body: JSON.stringify({ comment_id: commentId, reason }) }),

  // Influencers & Analytics
  getInfluencers: () => request("/api/influencers"),
  getInfluencerAnalytics: (infId) => request(`/api/influencers/${infId}/analytics`),

  // Moderation Console
  getModerationQueue: (status) => request(`/api/moderation/queue${status ? `?status_filter=${status}` : ""}`),
  executeAction: (commentId, action, reason) => request(`/api/moderation/comments/${commentId}/action`, { method: "POST", body: JSON.stringify({ action, reason }) }),
  getUserProfile: (userId) => request(`/api/moderation/users/${userId}/profile`),
  getAppeals: (status) => request(`/api/moderation/appeals${status ? `?status_filter=${status}` : ""}`),
  resolveAppeal: (appealId, action, notes) => request(`/api/moderation/appeals/${appealId}/resolve?action=${action}&notes=${encodeURIComponent(notes || "")}`, { method: "POST" }),
  getAuditLogs: () => request("/api/moderation/audit-logs"),

  // Federated Learning Console
  getFLStatus: () => request("/api/federated/status"),
  runFLRound: (payload) => request("/api/federated/run-round", { method: "POST", body: JSON.stringify(payload) }),
  getFLConvergence: () => request("/api/federated/convergence"),
  getFLProof: () => request("/api/federated/proof")
};
