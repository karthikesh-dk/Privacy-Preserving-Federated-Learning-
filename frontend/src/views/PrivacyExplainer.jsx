import React from "react";
import { 
  ShieldCheck, 
  Lock, 
  Cpu, 
  Layers, 
  HelpCircle, 
  EyeOff, 
  HeartHandshake, 
  CheckCircle2, 
  Sparkles,
  RefreshCw,
  AlertTriangle
} from "lucide-react";

export default function PrivacyExplainer() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-10">
      
      {/* Title & Introduction Banner */}
      <div className="text-center space-y-3">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs font-bold">
          <ShieldCheck className="w-4 h-4" />
          <span>EchoSafe Architecture</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          How Privacy-Preserving AI Works in EchoSafe
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 max-w-2xl mx-auto leading-relaxed">
          Traditional social networks upload and store every comment, private interaction, and behavioral profile on central corporate servers. EchoSafe flips this model using decentralized Federated Learning.
        </p>
      </div>

      {/* Comparison: Centralized vs EchoSafe */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Centralized (Old Model) */}
        <div className="bg-rose-50/60 border border-rose-200 rounded-2xl p-6 space-y-3">
          <div className="flex items-center space-x-2 text-rose-700 font-bold text-sm">
            <span>❌</span>
            <h3>Traditional Centralized Social Platforms</h3>
          </div>
          <ul className="space-y-2 text-xs text-rose-900">
            <li className="flex items-start space-x-2">
              <span className="text-rose-500 mt-0.5">•</span>
              <span>All user comments and messages are uploaded to a central cloud server.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-rose-500 mt-0.5">•</span>
              <span>Central databases are vulnerable to data breaches and insider surveillance.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-rose-500 mt-0.5">•</span>
              <span>Influencers have no custody over their own community's data.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-rose-500 mt-0.5">•</span>
              <span>Black-box moderation algorithms issue permanent bans without explainability.</span>
            </li>
          </ul>
        </div>

        {/* EchoSafe (Decentralized Federated Model) */}
        <div className="bg-emerald-50/60 border border-emerald-200 rounded-2xl p-6 space-y-3">
          <div className="flex items-center space-x-2 text-emerald-700 font-bold text-sm">
            <span>✅</span>
            <h3>EchoSafe Federated Architecture</h3>
          </div>
          <ul className="space-y-2 text-xs text-emerald-950">
            <li className="flex items-start space-x-2">
              <span className="text-emerald-600 mt-0.5">•</span>
              <span><strong>Data Locality:</strong> Comments stay strictly in creator-partitioned storage.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-emerald-600 mt-0.5">•</span>
              <span><strong>Federated Learning:</strong> The AI model travels to the data, computes gradients locally, and exchanges only mathematical model weights.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-emerald-600 mt-0.5">•</span>
              <span><strong>Cryptographic Masking:</strong> Secure Aggregation ensures the central server cannot inspect individual creator weights.</span>
            </li>
            <li className="flex items-start space-x-2">
              <span className="text-emerald-600 mt-0.5">•</span>
              <span><strong>Differential Privacy:</strong> Calibrated noise makes it mathematically impossible to reconstruct comments from model updates.</span>
            </li>
          </ul>
        </div>

      </div>

      {/* Deep-Dive Pillars */}
      <div className="space-y-6">
        <h2 className="text-lg font-extrabold text-slate-900">The 4 Pillars of EchoSafe Privacy & Safety</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Pillar 1: Federated Learning */}
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-3">
            <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center">
              <Cpu className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-sm text-slate-900">1. Federated Learning (FedAvg & FedProx)</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Instead of hoarding every creator's audience comments on one server, EchoSafe trains local models across 8 decentralized influencer nodes. The central coordinator collects only updated model parameter arrays and averages them using FedAvg (or FedProx to handle non-IID community skew).
            </p>
            <div className="p-2.5 rounded-xl bg-blue-50/50 border border-blue-100 text-[11px] text-blue-900 font-medium">
              💡 <strong>Guarantee:</strong> Zero raw text strings or user IDs ever leave the influencer client node.
            </div>
          </div>

          {/* Pillar 2: Secure Aggregation */}
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
              <Lock className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-sm text-slate-900">2. Secure Aggregation (SecAgg)</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Even weight updates might theoretically leak token frequencies through gradient inversion attacks. EchoSafe deploys Bonawitz pairwise zero-sum additive masking. Each creator node pairs with other nodes to generate complementary masks:
            </p>
            <div className="p-2.5 rounded-xl bg-indigo-50/50 border border-indigo-100 text-[11px] font-mono text-indigo-900">
              ∑ Masked_Weights = ∑ Weights + ∑ (Mask_ij - Mask_ji) = ∑ Weights
            </div>
            <p className="text-[11px] text-slate-500">
              The coordinator computes the exact aggregate model while learning <em>absolutely nothing</em> about any single creator's weights.
            </p>
          </div>

          {/* Pillar 3: Differential Privacy */}
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-3">
            <div className="w-10 h-10 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-sm text-slate-900">3. Bounded Differential Privacy (DP-FL)</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Differential privacy provides a rigorous mathematical proof: an observer looking at the final global model cannot tell whether any specific individual comment was included in the training dataset or not.
            </p>
            <div className="p-2.5 rounded-xl bg-purple-50/50 border border-purple-100 text-[11px] text-purple-900 font-medium">
              🛡️ <strong>L2-Norm Clipping (C=1.5) + Gaussian Perturbation (σ=0.8):</strong> Guarantees a bounded (ε, δ) privacy budget across rounds with provable plausible deniability.
            </div>
          </div>

          {/* Pillar 4: Graded Restorative Moderation */}
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-3">
            <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center">
              <HeartHandshake className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-sm text-slate-900">4. Graded & Restorative Interventions</h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Traditional platforms use automated, irreversible lifetime bans that alienate users and silence free speech. EchoSafe employs graded, time-limited interventions (Warnings, 15m Burst Cooldowns, Shadow Filters, and 48h Mutes).
            </p>
            <div className="p-2.5 rounded-xl bg-amber-50/50 border border-amber-100 text-[11px] text-amber-900 font-medium">
              ⚖️ <strong>Fair Right to Appeal:</strong> Every flagged decision is explainable with triggering words, and users can submit an appeal for human moderator review.
            </div>
          </div>

        </div>
      </div>

      {/* Safety Notice & Wellbeing Disclaimer */}
      <div className="bg-slate-100 rounded-2xl p-6 border border-slate-200 space-y-3 text-xs text-slate-600">
        <div className="flex items-center space-x-2 text-slate-900 font-bold">
          <AlertTriangle className="w-4 h-4 text-amber-600" />
          <span>Ethical AI & Wellbeing Disclaimer</span>
        </div>
        <p className="leading-relaxed">
          EchoSafe processes linguistic stress and toxicity patterns strictly to assist human community moderators.
          The <em>Wellbeing Signal</em> displayed to safety teams carries an explicit non-clinical disclaimer:
        </p>
        <blockquote className="border-l-4 border-amber-500 pl-3 py-1 font-semibold text-slate-800 italic">
          "Not a clinical diagnosis. For human safety team review only."
        </blockquote>
        <p className="text-[11px] text-slate-500">
          All seed data, influencer personas, and simulated comments in this demonstration are strictly synthetic and designed to prove mathematical privacy guarantees on real social media dynamics.
        </p>
      </div>

    </div>
  );
}