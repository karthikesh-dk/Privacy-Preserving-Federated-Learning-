import React, { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { 
  ShieldCheck, 
  Home, 
  BarChart3, 
  AlertTriangle, 
  Cpu, 
  HelpCircle, 
  UserCheck, 
  ChevronDown,
  LogOut
} from "lucide-react";

export default function Navbar({ activeTab, setActiveTab }) {
  const { user, switchDemoRole, logout } = useAuth();
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const demoRoles = [
    { key: "positive", label: "Follower (Positive User)", roleBadge: "Follower", color: "text-emerald-600" },
    { key: "critic", label: "Follower (Constructive Critic)", roleBadge: "Follower", color: "text-blue-600" },
    { key: "abuser", label: "Follower (Repeat Offender)", roleBadge: "Follower", color: "text-rose-600" },
    { key: "creator_gaming", label: "Influencer (Gaming Channel)", roleBadge: "Influencer", color: "text-purple-600" },
    { key: "creator_comedy", label: "Influencer (Comedy Skits)", roleBadge: "Influencer", color: "text-amber-600" },
    { key: "moderator", label: "Moderator (Safety Team)", roleBadge: "Moderator", color: "text-indigo-600" },
    { key: "admin", label: "Admin (FL Research Lead)", roleBadge: "Admin", color: "text-slate-900" }
  ];

  const tabs = [
    { id: "feed", label: "Home Feed", icon: Home },
    { id: "creator", label: "Creator Studio", icon: BarChart3 },
    { id: "moderation", label: "Moderator Console", icon: AlertTriangle, badge: "Queue" },
    { id: "federated", label: "FL Console", icon: Cpu, badge: "Admin" },
    { id: "privacy", label: "How Privacy Works", icon: HelpCircle }
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Brand */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab("feed")}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-md">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-lg text-slate-900 tracking-tight">EchoSafe</span>
                <span className="text-[10px] font-bold uppercase tracking-wider bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full border border-blue-200">
                  Federated AI
                </span>
              </div>
              <p className="text-[11px] text-slate-500 hidden sm:block">Privacy-Preserving Influencer Community</p>
            </div>
          </div>

          {/* Nav Tabs */}
          <nav className="hidden md:flex items-center space-x-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all duration-150 ${
                    isActive 
                      ? "bg-blue-600 text-white shadow-sm shadow-blue-200" 
                      : "text-slate-600 hover:text-slate-900 hover:bg-slate-100"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                  {tab.badge && (
                    <span className={`text-[9px] px-1.5 py-0.2 rounded-full uppercase font-bold ${
                      isActive ? "bg-white/20 text-white" : "bg-slate-200 text-slate-700"
                    }`}>
                      {tab.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Right: Role Switcher & User Profile */}
          <div className="flex items-center space-x-3">
            
            {/* Quick Demo Role Switcher Dropdown */}
            <div className="relative">
              <button
                onClick={() => setDropdownOpen(!dropdownOpen)}
                className="flex items-center space-x-2 px-3 py-1.5 rounded-xl border border-slate-200 bg-slate-50 hover:bg-slate-100 text-xs font-semibold text-slate-800 transition"
              >
                <UserCheck className="w-3.5 h-3.5 text-blue-600" />
                <span className="hidden sm:inline">Role:</span>
                <span className="capitalize text-blue-700 font-bold">{user?.role || "Follower"}</span>
                <ChevronDown className="w-3 h-3 text-slate-400" />
              </button>

              {dropdownOpen && (
                <div className="absolute right-0 mt-2 w-72 bg-white rounded-2xl shadow-xl border border-slate-200 p-2 z-50">
                  <div className="px-3 py-2 border-b border-slate-100 mb-1">
                    <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">1-Click Demo Account Switcher</p>
                    <p className="text-xs text-slate-500 mt-0.5">Test different permissions and moderation flows:</p>
                  </div>
                  <div className="space-y-0.5 max-h-72 overflow-y-auto">
                    {demoRoles.map((r) => (
                      <button
                        key={r.key}
                        onClick={() => {
                          switchDemoRole(r.key);
                          setDropdownOpen(false);
                        }}
                        className="w-full text-left px-3 py-2 rounded-xl text-xs hover:bg-slate-50 flex items-center justify-between group transition"
                      >
                        <div>
                          <div className="font-semibold text-slate-800">{r.label}</div>
                          <div className="text-[10px] text-slate-400">Demo switch to {r.roleBadge}</div>
                        </div>
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-slate-100 text-slate-700">
                          {r.roleBadge}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Current User Badge */}
            <div className="flex items-center space-x-2 pl-2 border-l border-slate-200">
              <img
                src={user?.avatar_url || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"}
                alt={user?.username}
                className="w-8 h-8 rounded-full border border-slate-200 object-cover"
              />
              <span className="text-xs font-bold text-slate-800 hidden lg:inline">{user?.username}</span>
            </div>

          </div>

        </div>

        {/* Mobile Navigation Row */}
        <div className="flex md:hidden overflow-x-auto space-x-2 py-2 border-t border-slate-100 scrollbar-none text-xs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg whitespace-nowrap text-xs font-semibold ${
                  isActive ? "bg-blue-600 text-white" : "text-slate-600 bg-slate-100"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

      </div>
    </header>
  );
}
