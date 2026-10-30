import React, { useState, useEffect } from "react";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";
import { 
  ShieldAlert, 
  CheckCircle, 
  AlertTriangle, 
  Clock, 
  Ban, 
  Eye, 
  EyeOff, 
  User, 
  HelpCircle, 
  FileText, 
  Activity, 
  TrendingUp, 
  Flame,
  Check,
  X,
  RefreshCw,
  Search
} from "lucide-react";

export default function ModeratorConsole() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState("queue"); // "queue" | "appeals" | "audit"
  const [queue, setQueue] = useState([]);
  const [appeals, setAppeals] = useState([]);
  const [auditLogs, setAuditLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [revealedIds, setRevealedIds] = useState({});
  const [actionInProgress, setActionInProgress] = useState({});
  
  // Profile modal state
  const [inspectUserId, setInspectUserId] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);

  // Appeal resolution modal
  const [resolvingAppeal, setResolvingAppeal] = useState(null);
  const [appealNotes, setAppealNotes] = useState("");

  useEffect(() => {
    fetchActiveData();
  }, [activeTab]);

  async function fetchActiveData() {
    setLoading(true);
    try {
      if (activeTab === "queue") {
        const q = await api.getModerationQueue("held_for_review");
        setQueue(q);
      } else if (activeTab === "appeals") {
        const a = await api.getAppeals();
        setAppeals(a);
      } else if (activeTab === "audit") {
        const l = await api.getAuditLogs();
        setAuditLogs(l);
      }
    } catch (err) {
      console.error("Failed to load moderation data:", err);
    } finally {
      setLoading(false);
    }
  }

  async function handleAction(commentId, action, reason) {
    setActionInProgress(prev => ({ ...prev, [commentId]: true }));
    try {
      await api.executeAction(commentId, action, reason || `Moderator action: ${action}`);
      // Remove from queue
      setQueue(queue.filter(c => c.id !== commentId));
    } catch (err) {
      alert(`Action failed: ${err.message}`);
    } finally {
      setActionInProgress(prev => ({ ...prev, [commentId]: false }));
    }
  }

  async function inspectUser(userId) {
    setInspectUserId(userId);
    setProfileLoading(true);
    try {
      const p = await api.getUserProfile(userId);
      setUserProfile(p);
    } catch (err) {
      alert(`Could not load profile: ${err.message}`);
      setInspectUserId(null);
    } finally {
      setProfileLoading(false);
    }
  }

  async function handleResolveAppeal(action) {
    if (!resolvingAppeal) return;
    try {
      await api.resolveAppeal(resolvingAppeal.id, action, appealNotes);
      setResolvingAppeal(null);
      setAppealNotes("");
      fetchActiveData();
    } catch (err) {
      alert(`Appeal action failed: ${err.message}`);
    }
  }

  function toggleReveal(id) {
    setRevealedIds(prev => ({ ...prev, [id]: !prev[id] }));
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      
      {/* Console Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
        <div>
          <div className="flex items-center space-x-2">
            <ShieldAlert className="w-5 h-5 text-indigo-600" />
            <h1 className="text-lg font-extrabold text-slate-900 tracking-tight">Community Safety & Moderation Console</h1>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Explainable AI decision triage, graded human-in-the-loop interventions, and restorative appeal workflows.
          </p>
        </div>

        {/* Section Tabs */}
        <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab("queue")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === "queue" ? "bg-white text-indigo-700 shadow-xs" : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Flagged Queue {queue.length > 0 && `(${queue.length})`}
          </button>
          <button
            onClick={() => setActiveTab("appeals")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === "appeals" ? "bg-white text-indigo-700 shadow-xs" : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Appeals {appeals.filter(a => a.status === "pending").length > 0 && `(${appeals.filter(a => a.status === "pending").length})`}
          </button>
          <button
            onClick={() => setActiveTab("audit")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              activeTab === "audit" ? "bg-white text-indigo-700 shadow-xs" : "text-slate-600 hover:text-slate-900"
            }`}
          >
            Audit Log
          </button>
          <button
            onClick={fetchActiveData}
            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-200 transition"
            title="Refresh"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {loading ? (
        <div className="text-center py-20 space-y-3">
          <div className="w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="text-xs text-slate-500 font-medium">Fetching moderation records...</p>
        </div>
      ) : activeTab === "queue" ? (
        
        /* 1. Flagged Moderation Queue */
        <div className="space-y-4">
          {queue.length === 0 ? (
            <div className="text-center py-16 bg-white rounded-2xl border border-slate-200">
              <CheckCircle className="w-8 h-8 text-emerald-500 mx-auto mb-2" />
              <h3 className="font-bold text-sm text-slate-800">Queue is Clear!</h3>
              <p className="text-xs text-slate-400 mt-1">No pending comments requiring human safety review right now.</p>
            </div>
          ) : (
            queue.map((item) => {
              const isRevealed = revealedIds[item.id];
              const isProcessing = actionInProgress[item.id];

              return (
                <div key={item.id} className="bg-white rounded-2xl border border-slate-200/90 shadow-sm p-5 space-y-4 hover:border-slate-300 transition">
                  
                  {/* Top Meta Header */}
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
                    <div className="flex items-center space-x-2">
                      <span className="font-bold text-xs text-slate-900">@{item.username}</span>
                      <button
                        onClick={() => inspectUser(item.user_id)}
                        className="px-2 py-0.5 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-bold text-[10px] flex items-center space-x-1 transition"
                      >
                        <User className="w-3 h-3" />
                        <span>Inspect Behavior Profile</span>
                      </button>
                    </div>

                    <div className="flex items-center space-x-2">
                      <span className={`text-[10px] font-extrabold uppercase px-2 py-0.5 rounded-md ${
                        item.severity === "severe" ? "bg-rose-100 text-rose-800 border border-rose-200" :
                        item.severity === "medium" ? "bg-amber-100 text-amber-800 border border-amber-200" :
                        "bg-blue-100 text-blue-800 border border-blue-200"
                      }`}>
                        Severity: {item.severity || "medium"}
                      </span>

                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-slate-100 text-slate-700">
                        Score: {(item.abuse_score * 100).toFixed(1)}%
                      </span>

                      <span className="text-[10px] text-slate-400">
                        {item.created_at ? new Date(item.created_at).toLocaleTimeString() : ""}
                      </span>
                    </div>
                  </div>

                  {/* Masked / Revealed Text Display */}
                  <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/70 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                        {isRevealed ? "Raw Comment (Unmasked)" : "Masked Safe Preview"}
                      </span>
                      <button
                        onClick={() => toggleReveal(item.id)}
                        className="text-[11px] font-bold text-indigo-600 hover:text-indigo-800 flex items-center space-x-1"
                      >
                        {isRevealed ? (
                          <>
                            <EyeOff className="w-3.5 h-3.5" />
                            <span>Mask Again</span>
                          </>
                        ) : (
                          <>
                            <Eye className="w-3.5 h-3.5" />
                            <span>Click to Reveal (Moderator Only)</span>
                          </>
                        )}
                      </button>
                    </div>

                    <p className={`text-xs leading-relaxed ${isRevealed ? "text-slate-900 font-medium" : "text-slate-500 filter blur-xs select-none"}`}>
                      {isRevealed ? item.text : (item.masked_text || item.text)}
                    </p>

                    {/* Explainability Keywords */}
                    {item.explainability_keywords && item.explainability_keywords.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1 pt-1">
                        <span className="text-[10px] font-semibold text-rose-600">Top Explanatory Keywords:</span>
                        {item.explainability_keywords.map((kw, i) => (
                          <span key={i} className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-rose-50 text-rose-700 border border-rose-200">
                            {kw}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Policy Graded Action Buttons */}
                  <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                    <div className="text-[11px] text-slate-500">
                      <span className="font-semibold text-slate-700">Flag Reason: </span>
                      <span>{item.reason || "Automatic threshold breach"}</span>
                    </div>

                    <div className="flex flex-wrap items-center gap-1.5">
                      <button
                        disabled={isProcessing}
                        onClick={() => handleAction(item.id, "approve", "Approved by human moderator as benign/permissible")}
                        className="px-3 py-1.5 rounded-xl bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-300 text-xs font-bold transition flex items-center space-x-1"
                      >
                        <CheckCircle className="w-3.5 h-3.5" />
                        <span>Approve</span>
                      </button>

                      <button
                        disabled={isProcessing}
                        onClick={() => handleAction(item.id, "warn", "Warning issued: Respectful language guidelines")}
                        className="px-3 py-1.5 rounded-xl bg-amber-50 hover:bg-amber-100 text-amber-700 border border-amber-300 text-xs font-bold transition flex items-center space-x-1"
                      >
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>Warn</span>
                      </button>

                      <button
                        disabled={isProcessing}
                        onClick={() => handleAction(item.id, "slow_mode_15m", "15-minute burst cooling mode")}
                        className="px-3 py-1.5 rounded-xl bg-purple-50 hover:bg-purple-100 text-purple-700 border border-purple-300 text-xs font-bold transition flex items-center space-x-1"
                      >
                        <Clock className="w-3.5 h-3.5" />
                        <span>15m Slow Mode</span>
                      </button>

                      <button
                        disabled={isProcessing}
                        onClick={() => handleAction(item.id, "shadow_filter", "Hidden from public creator feed")}
                        className="px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 text-xs font-bold transition flex items-center space-x-1"
                      >
                        <EyeOff className="w-3.5 h-3.5" />
                        <span>Shadow Filter</span>
                      </button>

                      <button
                        disabled={isProcessing}
                        onClick={() => handleAction(item.id, "mute_48h", "48-hour temporary quiet period")}
                        className="px-3 py-1.5 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-300 text-xs font-bold transition flex items-center space-x-1"
                      >
                        <Ban className="w-3.5 h-3.5" />
                        <span>48h Mute</span>
                      </button>
                    </div>
                  </div>

                </div>
              );
            })
          )}
        </div>

      ) : activeTab === "appeals" ? (
        
        /* 2. Appeals Resolution Center */
        <div className="space-y-4">
          {appeals.length === 0 ? (
            <div className="text-center py-16 bg-white rounded-2xl border border-slate-200">
              <CheckCircle className="w-8 h-8 text-indigo-500 mx-auto mb-2" />
              <h3 className="font-bold text-sm text-slate-800">No Pending Appeals</h3>
              <p className="text-xs text-slate-400 mt-1">All user appeals have been processed.</p>
            </div>
          ) : (
            appeals.map((app) => (
              <div key={app.id} className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-3">
                <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-xs text-slate-900">User Appeal #{app.id.substring(0, 8)}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      app.status === "pending" ? "bg-amber-100 text-amber-800" :
                      app.status === "accepted" ? "bg-emerald-100 text-emerald-800" :
                      "bg-rose-100 text-rose-800"
                    }`}>
                      {app.status.toUpperCase()}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400">{new Date(app.created_at).toLocaleString()}</span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="font-bold text-[10px] uppercase text-slate-400 block mb-1">Original Flagged Content</span>
                    <p className="text-slate-800">{app.comment_text || "(Text held under policy)"}</p>
                  </div>
                  <div className="p-3 rounded-xl bg-indigo-50/50 border border-indigo-100">
                    <span className="font-bold text-[10px] uppercase text-indigo-600 block mb-1">User's Appeal Defense</span>
                    <p className="text-indigo-950">{app.reason}</p>
                  </div>
                </div>

                {app.status === "pending" && (
                  <div className="flex items-center justify-end space-x-2 pt-2">
                    <button
                      onClick={() => {
                        setResolvingAppeal(app);
                        setAppealNotes("Appeal accepted after review of constructive context.");
                      }}
                      className="px-3.5 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold transition flex items-center space-x-1"
                    >
                      <Check className="w-3.5 h-3.5" />
                      <span>Accept & Restore Comment</span>
                    </button>
                    <button
                      onClick={() => {
                        setResolvingAppeal(app);
                        setAppealNotes("Appeal rejected: Content repeatedly breaches community respect norms.");
                      }}
                      className="px-3.5 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold transition flex items-center space-x-1"
                    >
                      <X className="w-3.5 h-3.5" />
                      <span>Reject Appeal</span>
                    </button>
                  </div>
                )}
              </div>
            ))
          )}
        </div>

      ) : (
        
        /* 3. Decision Audit Log */
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-sm text-slate-900">Immutable Moderation Audit Log</h3>
            <span className="text-xs text-slate-400">Total events recorded: {auditLogs.length}</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-100 text-slate-400 uppercase tracking-wider text-[10px]">
                  <th className="pb-3 font-semibold">Timestamp</th>
                  <th className="pb-3 font-semibold">Action</th>
                  <th className="pb-3 font-semibold">Target User</th>
                  <th className="pb-3 font-semibold">Actor / System</th>
                  <th className="pb-3 font-semibold">Justification / Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {auditLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-50/70 transition">
                    <td className="py-2.5 text-slate-400 text-[10px]">
                      {new Date(log.created_at).toLocaleString()}
                    </td>
                    <td className="py-2.5">
                      <span className="font-bold text-[10px] px-2 py-0.5 rounded-md bg-slate-100 text-slate-800">
                        {log.action}
                      </span>
                    </td>
                    <td className="py-2.5 font-bold text-slate-800">
                      @{log.target_user_id}
                    </td>
                    <td className="py-2.5 text-slate-600 text-[11px]">
                      {log.actor_user_id || "Automated Policy Engine"}
                    </td>
                    <td className="py-2.5 text-slate-600 text-[11px] max-w-xs truncate">
                      {log.reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* User Behavioral Profile Modal */}
      {inspectUserId && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-xl w-full p-6 shadow-2xl border border-slate-200 space-y-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center space-x-2">
                <Activity className="w-5 h-5 text-indigo-600" />
                <h3 className="font-bold text-base text-slate-900">User Behavioral Profile</h3>
              </div>
              <button
                onClick={() => setInspectUserId(null)}
                className="text-slate-400 hover:text-slate-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {profileLoading || !userProfile ? (
              <div className="py-12 text-center text-xs text-slate-500">
                Loading behavioral signals...
              </div>
            ) : (
              <div className="space-y-4 text-xs">
                
                {/* User Header */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <div>
                    <span className="text-sm font-extrabold text-slate-900">@{userProfile.username}</span>
                    <span className="text-[11px] text-slate-500 block">Account ID: {userProfile.user_id}</span>
                  </div>
                  <div className="text-right">
                    <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold ${
                      userProfile.is_restricted ? "bg-rose-100 text-rose-800" : "bg-emerald-100 text-emerald-800"
                    }`}>
                      {userProfile.is_restricted ? `Restricted (${userProfile.restriction_type})` : "Standard Access"}
                    </span>
                    {userProfile.cooldown_seconds_remaining > 0 && (
                      <span className="text-[10px] text-rose-600 block mt-0.5">
                        Cooldown: {userProfile.cooldown_seconds_remaining}s remaining
                      </span>
                    )}
                  </div>
                </div>

                {/* Behavioral Metrics Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase block">Total Comments</span>
                    <span className="text-lg font-bold text-slate-900">{userProfile.total_comments}</span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase block">Negative Share</span>
                    <span className="text-lg font-bold text-rose-600">{(userProfile.negative_ratio * 100).toFixed(1)}%</span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase block">Repeat Offenses</span>
                    <span className="text-lg font-bold text-rose-700">{userProfile.repeat_offence_count}</span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase block">Escalation Slope</span>
                    <span className="text-lg font-bold text-indigo-700">{userProfile.escalation_slope > 0 ? `+${userProfile.escalation_slope.toFixed(2)}` : userProfile.escalation_slope.toFixed(2)}</span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase block">Burst Count</span>
                    <span className="text-lg font-bold text-amber-600">{userProfile.burst_count}</span>
                  </div>

                  {/* Wellbeing Signal Card */}
                  <div className="p-3 rounded-xl bg-indigo-50 border border-indigo-200">
                    <span className="text-[10px] font-semibold text-indigo-700 uppercase block">Wellbeing Signal</span>
                    <span className={`text-sm font-extrabold uppercase ${
                      userProfile.wellbeing_signal === "high" ? "text-rose-600" :
                      userProfile.wellbeing_signal === "medium" ? "text-amber-600" :
                      "text-emerald-600"
                    }`}>
                      {userProfile.wellbeing_signal} Tension
                    </span>
                  </div>
                </div>

                {/* Preferred Post Types */}
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                  <span className="font-bold text-[10px] uppercase text-slate-400 block">Engagement Preferences:</span>
                  <div className="flex flex-wrap gap-2 text-[11px]">
                    <span className="text-slate-600">
                      <strong>Comments:</strong> {userProfile.preferred_post_types_comments?.join(", ") || "Diverse"}
                    </span>
                    <span className="text-slate-400">|</span>
                    <span className="text-slate-600">
                      <strong>Likes:</strong> {userProfile.preferred_post_types_likes?.join(", ") || "Diverse"}
                    </span>
                  </div>
                </div>

                {/* MANDATORY NON-CLINICAL DISCLAIMER */}
                <div className="p-3.5 rounded-xl bg-amber-50/80 border border-amber-300 text-amber-900 space-y-1">
                  <div className="flex items-center space-x-1.5 font-bold text-[11px] text-amber-800">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                    <span>Non-Clinical Human Review Notice</span>
                  </div>
                  <p className="text-[11px] leading-relaxed text-amber-800 italic">
                    "{userProfile.wellbeing_disclaimer || "Not a clinical diagnosis. For human safety team review only."}"
                  </p>
                  <p className="text-[10px] text-amber-700">
                    Signals reflect linguistic agitation and posting frequency patterns to guide restorative, empathetic moderation rather than punitive measures.
                  </p>
                </div>

                <div className="flex justify-end pt-2">
                  <button
                    onClick={() => setInspectUserId(null)}
                    className="px-4 py-2 rounded-xl bg-slate-900 text-white font-bold text-xs hover:bg-slate-800"
                  >
                    Close Profile
                  </button>
                </div>

              </div>
            )}
          </div>
        </div>
      )}

      {/* Appeal Resolution Modal */}
      {resolvingAppeal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-200 space-y-4">
            <h3 className="font-bold text-base text-slate-900">Resolve Appeal</h3>
            <p className="text-xs text-slate-600">
              Provide justification note for the decision audit log:
            </p>

            <textarea
              rows={3}
              value={appealNotes}
              onChange={(e) => setAppealNotes(e.target.value)}
              className="w-full text-xs p-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              placeholder="e.g. Reviewed context; comment adheres to respectful discourse guidelines."
            />

            <div className="flex items-center justify-end space-x-2 pt-2">
              <button
                onClick={() => setResolvingAppeal(null)}
                className="px-3 py-1.5 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100"
              >
                Cancel
              </button>
              <button
                onClick={() => handleResolveAppeal("accept")}
                className="px-3.5 py-1.5 rounded-xl text-xs font-bold bg-emerald-600 text-white hover:bg-emerald-700"
              >
                Accept Appeal
              </button>
              <button
                onClick={() => handleResolveAppeal("reject")}
                className="px-3.5 py-1.5 rounded-xl text-xs font-bold bg-rose-600 text-white hover:bg-rose-700"
              >
                Reject Appeal
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}