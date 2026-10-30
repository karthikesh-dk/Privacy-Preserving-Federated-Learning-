import React, { useState, useEffect } from "react";
import { api } from "../api";
import { 
  BarChart3, 
  ShieldCheck, 
  AlertOctagon, 
  TrendingUp, 
  Users, 
  MessageCircle, 
  Sparkles,
  ShieldAlert,
  Flame,
  CheckCircle2,
  RefreshCw
} from "lucide-react";

import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title
} from "chart.js";
import { Doughnut, Bar } from "react-chartjs-2";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
  Title
);

export default function InfluencerStudio() {
  const [influencers, setInfluencers] = useState([]);
  const [selectedInfId, setSelectedInfId] = useState("");
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadInfluencers() {
      try {
        const list = await api.getInfluencers();
        setInfluencers(list);
        if (list.length > 0) {
          setSelectedInfId(list[0].id);
        }
      } catch (e) {
        console.error("Failed to load creators list:", e);
      }
    }
    loadInfluencers();
  }, []);

  useEffect(() => {
    if (selectedInfId) {
      loadAnalytics(selectedInfId);
    }
  }, [selectedInfId]);

  async function loadAnalytics(id) {
    setLoading(true);
    try {
      const data = await api.getInfluencerAnalytics(id);
      setAnalytics(data);
    } catch (err) {
      console.error("Failed to load creator analytics:", err);
    } finally {
      setLoading(false);
    }
  }

  // Sentiment Doughnut Chart Data
  const sentimentData = {
    labels: ["Positive & Supportive", "Neutral Discourse", "Constructive Critique", "Abusive / Hostile"],
    datasets: [
      {
        data: [
          analytics?.sentiment_breakdown?.positive || 0,
          analytics?.sentiment_breakdown?.neutral || 0,
          analytics?.sentiment_breakdown?.criticism || 0,
          analytics?.sentiment_breakdown?.abusive || 0
        ],
        backgroundColor: [
          "rgba(16, 185, 129, 0.85)", // emerald
          "rgba(148, 163, 184, 0.85)", // slate
          "rgba(59, 130, 246, 0.85)",  // blue
          "rgba(244, 63, 94, 0.85)"    // rose
        ],
        borderColor: ["#10b981", "#94a3b8", "#3b82f6", "#f43f5e"],
        borderWidth: 2
      }
    ]
  };

  // Post Type Abuse Rates Bar Chart
  const postTypes = Object.keys(analytics?.post_type_abuse_rates || {});
  const postAbuseData = {
    labels: postTypes.map(pt => pt.toUpperCase()),
    datasets: [
      {
        label: "Abuse Rate (%)",
        data: postTypes.map(pt => analytics?.post_type_abuse_rates[pt] || 0),
        backgroundColor: "rgba(239, 68, 68, 0.75)",
        borderColor: "#ef4444",
        borderWidth: 1.5,
        borderRadius: 8
      }
    ]
  };

  const healthScore = analytics?.community_health_score ?? 85;
  const isHealthy = healthScore >= 75;
  const isWarning = healthScore < 75 && healthScore >= 50;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      
      {/* Header & Influencer Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
        <div>
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5 text-indigo-600" />
            <h1 className="text-lg font-extrabold text-slate-900 tracking-tight">Creator Safety & Analytics Studio</h1>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Real-time community health monitoring, post-format risk analysis, and decentralized protection metrics.
          </p>
        </div>

        {/* Creator Channel Selector */}
        <div className="flex items-center space-x-2">
          <label className="text-xs font-bold text-slate-600 whitespace-nowrap">Channel:</label>
          <select
            value={selectedInfId}
            onChange={(e) => setSelectedInfId(e.target.value)}
            className="text-xs font-semibold bg-slate-50 border border-slate-300 rounded-xl px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-500 shadow-sm"
          >
            {influencers.map((inf) => (
              <option key={inf.id} value={inf.id}>
                {inf.name} ({inf.niche})
              </option>
            ))}
          </select>
          <button
            onClick={() => loadAnalytics(selectedInfId)}
            className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-600 transition"
            title="Refresh Data"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-20 space-y-3">
          <div className="w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="text-xs text-slate-500 font-medium">Crunching creator community metrics...</p>
        </div>
      ) : (
        <>
          {/* Surge Warning Alert Banner (If Raid or Velocity Spike) */}
          {(analytics?.is_raid_detected || analytics?.toxicity_velocity > 0.15) && (
            <div className="bg-rose-50 border-2 border-rose-400 rounded-2xl p-4 flex items-start space-x-3 text-rose-950 shadow-md animate-bounce duration-500">
              <AlertOctagon className="w-6 h-6 text-rose-600 mt-0.5 flex-shrink-0" />
              <div>
                <h4 className="font-extrabold text-sm text-rose-900">Harassment Surge Warning Detected!</h4>
                <p className="text-xs text-rose-800 mt-0.5 leading-relaxed">
                  EchoSafe's burst detector identified an abnormal surge in negative interactions. Automatic slow-mode
                  and safety filters have been recommended to protect creator mental wellbeing.
                </p>
                <p className="mt-1 text-[11px] font-bold text-rose-700">
                  Recommended action: {analytics?.protective_recommendation || "Activate 15m comment cooldown"}
                </p>
              </div>
            </div>
          )}

          {/* Top Key Metrics Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            
            {/* Health Score Card */}
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-slate-400 uppercase tracking-wider">
                <span>Community Health</span>
                <ShieldCheck className={`w-4 h-4 ${isHealthy ? "text-emerald-500" : isWarning ? "text-amber-500" : "text-rose-500"}`} />
              </div>
              <div className="flex items-baseline space-x-2">
                <span className={`text-3xl font-extrabold ${isHealthy ? "text-emerald-600" : isWarning ? "text-amber-600" : "text-rose-600"}`}>
                  {healthScore.toFixed(0)}
                </span>
                <span className="text-xs text-slate-400 font-bold">/ 100</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                <div 
                  className={`h-full rounded-full ${isHealthy ? "bg-emerald-500" : isWarning ? "bg-amber-500" : "bg-rose-500"}`}
                  style={{ width: `${Math.min(100, Math.max(0, healthScore))}%` }}
                />
              </div>
              <span className="text-[11px] font-semibold text-slate-500 block">
                {isHealthy ? "Healthy & Supportive" : isWarning ? "Moderate Contention" : "Elevated Toxicity"}
              </span>
            </div>

            {/* Toxicity Velocity */}
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-slate-400 uppercase tracking-wider">
                <span>Toxicity Velocity</span>
                <TrendingUp className="w-4 h-4 text-rose-500" />
              </div>
              <div className="flex items-baseline space-x-1">
                <span className="text-3xl font-extrabold text-slate-900">
                  {(analytics?.toxicity_velocity || 0).toFixed(2)}
                </span>
                <span className="text-xs text-slate-400 font-bold">fl/min</span>
              </div>
              <span className="text-[11px] font-semibold text-slate-500 block">
                {analytics?.toxicity_velocity > 0.1 ? "Surge in hostile velocity" : "Standard baseline engagement"}
              </span>
            </div>

            {/* Total Posts */}
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-slate-400 uppercase tracking-wider">
                <span>Total Posts</span>
                <MessageCircle className="w-4 h-4 text-blue-500" />
              </div>
              <span className="text-3xl font-extrabold text-slate-900 block">
                {analytics?.total_posts || 0}
              </span>
              <span className="text-[11px] font-semibold text-slate-500 block">
                Published across 5 formats
              </span>
            </div>

            {/* Total Comments */}
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-slate-400 uppercase tracking-wider">
                <span>Audience Comments</span>
                <Users className="w-4 h-4 text-purple-500" />
              </div>
              <span className="text-3xl font-extrabold text-slate-900 block">
                {analytics?.total_comments || 0}
              </span>
              <span className="text-[11px] font-semibold text-slate-500 block">
                Processed locally by on-device model
              </span>
            </div>

          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Sentiment Breakdown Doughnut Chart */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
              <div>
                <h3 className="font-bold text-sm text-slate-900">Comment Sentiment Distribution</h3>
                <p className="text-xs text-slate-400">Classified live via local federated inference model</p>
              </div>
              <div className="h-64 flex items-center justify-center">
                <Doughnut
                  data={sentimentData}
                  options={{
                    maintainAspectRatio: false,
                    plugins: {
                      legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } }
                    }
                  }}
                />
              </div>
            </div>

            {/* Post Format Abuse Rates */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
              <div>
                <h3 className="font-bold text-sm text-slate-900">Which Post Types Attract Negativity?</h3>
                <p className="text-xs text-slate-400">Abusive comment proportion across different content formats</p>
              </div>
              <div className="h-64">
                <Bar
                  data={postAbuseData}
                  options={{
                    maintainAspectRatio: false,
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 50,
                        ticks: { callback: (v) => `${v}%`, font: { size: 10 } }
                      },
                      x: { ticks: { font: { size: 10 } } }
                    },
                    plugins: {
                      legend: { display: false }
                    }
                  }}
                />
              </div>
            </div>

          </div>

          {/* Top Repeat Offenders Table & Community Shield */}
          <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-bold text-sm text-slate-900">Top Repeat Targeters & Flagged Accounts</h3>
                <p className="text-xs text-slate-400">
                  Accounts with repeated hostility targeting this creator's community:
                </p>
              </div>
              <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-600">
                Graded Protection
              </span>
            </div>

            {analytics?.top_repeat_offenders?.length === 0 ? (
              <div className="text-center py-8 text-xs text-slate-500 font-medium">
                No repeat offenders detected! Community interactions are currently civil.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-slate-100 text-slate-400 uppercase tracking-wider text-[10px]">
                      <th className="pb-3 font-semibold">User</th>
                      <th className="pb-3 font-semibold">Repeat Hostility Count</th>
                      <th className="pb-3 font-semibold">Policy Status</th>
                      <th className="pb-3 font-semibold">Intervention</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {analytics.top_repeat_offenders.map((off, idx) => (
                      <tr key={idx} className="hover:bg-slate-50/70 transition">
                        <td className="py-3 font-bold text-slate-800">
                          @{off.username}
                        </td>
                        <td className="py-3 font-semibold text-rose-600">
                          {off.repeat_offences_here} incidents
                        </td>
                        <td className="py-3">
                          {off.is_restricted ? (
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-100 text-rose-800">
                              Restricted ({off.restriction_type})
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-100 text-amber-800">
                              Warning Phase
                            </span>
                          )}
                        </td>
                        <td className="py-3 text-slate-500 text-[11px]">
                          {off.is_restricted ? "Under automatic cooldown" : "Monitored by policy engine"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Protective Advice Banner */}
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-2xl p-5 border border-indigo-100 flex items-start space-x-3 text-xs">
            <Sparkles className="w-5 h-5 text-indigo-600 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-bold text-indigo-950">Creator Wellbeing Advisory</h4>
              <p className="text-indigo-800 mt-1 leading-relaxed">
                EchoSafe ensures creators don't have to face psychological burnout from online hate. Abusive content
                is filtered and queued before reaching your direct notifications, and federated learning updates the
                protection boundary without your private audience messages ever leaving your device.
              </p>
            </div>
          </div>
        </>
      )}

    </div>
  );
}