import React, { useState } from "react";
import { AuthProvider, useAuth } from "./context/AuthContext";
import Navbar from "./components/Navbar";
import DemoSimulatorBar from "./components/DemoSimulatorBar";
import FeedView from "./views/FeedView";
import InfluencerStudio from "./views/InfluencerStudio";
import ModeratorConsole from "./views/ModeratorConsole";
import FederatedConsole from "./views/FederatedConsole";
import PrivacyExplainer from "./views/PrivacyExplainer";
import { ShieldCheck, Lock, Activity, Heart } from "lucide-react";

function MainContent() {
  const [activeTab, setActiveTab] = useState("feed");
  const [refreshKey, setRefreshKey] = useState(0);

  function handleCommentInjected() {
    setRefreshKey(prev => prev + 1);
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      
      {/* Top Navbar */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Simulator Bar for 1-Click Interactive Demo Scenarios */}
      <DemoSimulatorBar onCommentInjected={handleCommentInjected} />

      {/* Primary Dynamic View */}
      <main className="flex-1 pb-16">
        {activeTab === "feed" && <FeedView key={refreshKey} />}
        {activeTab === "creator" && <InfluencerStudio key={refreshKey} />}
        {activeTab === "moderation" && <ModeratorConsole key={refreshKey} />}
        {activeTab === "federated" && <FederatedConsole key={refreshKey} />}
        {activeTab === "privacy" && <PrivacyExplainer />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 px-4 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          
          <div className="flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-blue-600" />
            <span className="font-extrabold text-slate-900">EchoSafe</span>
            <span className="text-slate-400">|</span>
            <span className="font-semibold text-slate-700">
              Privacy-Preserving Federated Learning for Social Media Influencer
            </span>
          </div>

          <div className="flex items-center space-x-4 text-[11px] text-slate-400">
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block animate-pulse"></span>
              <span>Federated Mesh: 8 Nodes Online</span>
            </span>
            <span>•</span>
            <span>Zero Raw Data Leakage</span>
            <span>•</span>
            <span>Flower FedAvg & SecAgg</span>
          </div>

        </div>
      </footer>

    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainContent />
    </AuthProvider>
  );
}