import React, { useState } from "react";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";
import { Zap, CheckCircle2, AlertCircle, ShieldAlert, Sparkles, MessageSquare } from "lucide-react";

export default function DemoSimulatorBar({ onCommentInjected }) {
  const { switchDemoRole } = useAuth();
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);

  function showToast(title, message, type = "success") {
    setToast({ title, message, type });
    setTimeout(() => setToast(null), 4500);
  }

  async function injectScenario(type) {
    setLoading(true);
    try {
      if (type === "positive") {
        await switchDemoRole("positive");
        const res = await api.postComment({
          post_id: "post_0001",
          text: "Your editing and tutorial breakdown is fantastic! Learned so much from this."
        });
        showToast(
          "Praise Scenario Injected",
          `Status: ${res.status.toUpperCase()} (${res.label}) — Benign positive comment approved.`,
          "success"
        );
      } else if (type === "criticism") {
        await switchDemoRole("critic");
        const res = await api.postComment({
          post_id: "post_0001",
          text: "I disagree with your benchmark conclusions. The test methodology seems flawed."
        });
        showToast(
          "Constructive Criticism Injected",
          `Status: ${res.status.toUpperCase()} (${res.label}) — Community Free Speech Protected (Zero Censorship).`,
          "info"
        );
      } else if (type === "mild_warning") {
        await switchDemoRole("positive");
        const res = await api.postComment({
          post_id: "post_0001",
          text: "This product design is completely brainless and trash."
        });
        showToast(
          "Warning Flow Injected",
          `Status: ${res.status.toUpperCase()} — Published, but warning notification sent to author!`,
          "warning"
        );
      } else if (type === "escalating") {
        await switchDemoRole("abuser");
        const res = await api.postComment({
          post_id: "post_0001",
          text: "You are an incompetent clown, stop making garbage videos and leave."
        });
        showToast(
          "Escalation Flow Injected",
          `Status: ${res.status.toUpperCase()} — Flagged by behavioral escalation slope and held for moderator review!`,
          "danger"
        );
      } else if (type === "burst_spam") {
        await switchDemoRole("abuser");
        // Fire rapid comments
        await api.postComment({ post_id: "post_0001", text: "Trash creator clown 1!" });
        const res2 = await api.postComment({ post_id: "post_0001", text: "Trash creator clown 2!" });
        showToast(
          "Burst Attack Injected",
          `Status: ${res2.status.toUpperCase()} — Rapid bursts (<5m) detected! 15-minute slow mode cooldown enforced!`,
          "danger"
        );
      }

      if (onCommentInjected) onCommentInjected();
    } catch (err) {
      showToast("Injection Status", err.message, "danger");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white border-b border-indigo-900/50 py-2.5 px-4 sticky top-16 z-40 shadow-md">
      <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3 text-xs">
        
        <div className="flex items-center space-x-2">
          <span className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <Zap className="w-4 h-4" />
          </span>
          <div>
            <span className="font-bold text-slate-100 uppercase tracking-wider text-[11px] block">Live Moderator & Policy Simulator</span>
            <span className="text-[10px] text-slate-400">1-click scripted injection scenarios to demonstrate model & behavioral pipeline:</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            disabled={loading}
            onClick={() => injectScenario("positive")}
            className="px-2.5 py-1.5 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 font-semibold transition flex items-center space-x-1"
          >
            <span>✨</span>
            <span>1. Praise (Positive)</span>
          </button>

          <button
            disabled={loading}
            onClick={() => injectScenario("criticism")}
            className="px-2.5 py-1.5 rounded-lg bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 border border-blue-500/40 font-semibold transition flex items-center space-x-1"
          >
            <span>💬</span>
            <span>2. Fair Critique (Protected)</span>
          </button>

          <button
            disabled={loading}
            onClick={() => injectScenario("mild_warning")}
            className="px-2.5 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 font-semibold transition flex items-center space-x-1"
          >
            <span>⚠️</span>
            <span>3. Borderline (Warning)</span>
          </button>

          <button
            disabled={loading}
            onClick={() => injectScenario("escalating")}
            className="px-2.5 py-1.5 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 border border-purple-500/40 font-semibold transition flex items-center space-x-1"
          >
            <span>📈</span>
            <span>4. Escalating Toxicity</span>
          </button>

          <button
            disabled={loading}
            onClick={() => injectScenario("burst_spam")}
            className="px-2.5 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 font-semibold transition flex items-center space-x-1"
          >
            <span>🚨</span>
            <span>5. Burst Attack (Mute)</span>
          </button>
        </div>

      </div>

      {/* Floating Toast Notification */}
      {toast && (
        <div className={`fixed bottom-6 right-6 max-w-md p-4 rounded-2xl shadow-2xl border text-xs z-50 animate-bounce duration-300 ${
          toast.type === "success" ? "bg-emerald-950/95 border-emerald-500 text-emerald-100" :
          toast.type === "warning" ? "bg-amber-950/95 border-amber-500 text-amber-100" :
          toast.type === "info" ? "bg-blue-950/95 border-blue-500 text-blue-100" :
          "bg-rose-950/95 border-rose-500 text-rose-100"
        }`}>
          <div className="flex items-start space-x-3">
            <span className="text-lg">
              {toast.type === "success" ? "✅" : toast.type === "warning" ? "⚠️" : toast.type === "info" ? "💬" : "🚨"}
            </span>
            <div>
              <h4 className="font-bold text-sm">{toast.title}</h4>
              <p className="mt-1 text-slate-300 leading-relaxed">{toast.message}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
