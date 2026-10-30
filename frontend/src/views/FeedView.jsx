import React, { useState, useEffect } from "react";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";
import { 
  Heart, 
  MessageSquare, 
  Share2, 
  Bookmark, 
  Send, 
  ShieldAlert, 
  AlertTriangle, 
  Eye, 
  EyeOff, 
  Sparkles, 
  PlusCircle, 
  Filter,
  CheckCircle,
  HelpCircle,
  UserCheck
} from "lucide-react";

export default function FeedView() {
  const { user } = useAuth();
  const [posts, setPosts] = useState([]);
  const [activeFilter, setActiveFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [commentInputs, setCommentInputs] = useState({});
  const [submittingComment, setSubmittingComment] = useState({});
  const [expandedComments, setExpandedComments] = useState({});
  const [revealedAbuse, setRevealedAbuse] = useState({});
  const [userProfile, setUserProfile] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newPost, setNewPost] = useState({
    title: "",
    content: "",
    post_type: "opinion",
    media_url: ""
  });
  const [appealModal, setAppealModal] = useState(null);
  const [appealReason, setAppealReason] = useState("");
  const [feedbackNotice, setFeedbackNotice] = useState(null);

  const postTypeFilters = [
    { id: "all", label: "All Posts" },
    { id: "opinion", label: "Opinions" },
    { id: "reel", label: "Reels" },
    { id: "image", label: "Images" },
    { id: "video", label: "Videos" },
    { id: "live", label: "Live Debates" },
  ];

  useEffect(() => {
    fetchPosts();
    if (user?.id) {
      fetchUserProfile();
    }
  }, [activeFilter, user?.id]);

  async function fetchPosts() {
    setLoading(true);
    try {
      const data = await api.getFeed(activeFilter === "all" ? null : activeFilter);
      setPosts(data);
    } catch (err) {
      console.error("Failed to load feed:", err);
    } finally {
      setLoading(false);
    }
  }

  async function fetchUserProfile() {
    try {
      const p = await api.getUserProfile(user.id);
      setUserProfile(p);
    } catch (e) {
      // Ignore if not moderator permission or new user
    }
  }

  async function handleToggleLike(postId) {
    try {
      const res = await api.toggleLike(postId);
      setPosts(posts.map(p => {
        if (p.id === postId) {
          return {
            ...p,
            likes_count: res.liked ? p.likes_count + 1 : Math.max(0, p.likes_count - 1),
            has_liked: res.liked
          };
        }
        return p;
      }));
    } catch (err) {
      console.error("Like failed:", err);
    }
  }

  async function handlePostComment(postId) {
    const text = commentInputs[postId]?.trim();
    if (!text) return;

    setSubmittingComment(prev => ({ ...prev, [postId]: true }));
    try {
      const res = await api.postComment({ post_id: postId, text });
      
      // Clear input
      setCommentInputs(prev => ({ ...prev, [postId]: "" }));
      
      // Feedback banner
      if (res.status === "published") {
        setFeedbackNotice({
          type: "success",
          msg: "Your comment was published instantly!"
        });
      } else if (res.status === "warning") {
        setFeedbackNotice({
          type: "warning",
          msg: `Comment posted with warning: ${res.reason || "Please keep community discourse respectful."}`
        });
      } else if (res.status === "held_for_review") {
        setFeedbackNotice({
          type: "danger",
          msg: `Comment held for moderation review: ${res.reason || "Potential policy violation detected."}`
        });
      }

      setTimeout(() => setFeedbackNotice(null), 5000);

      // Refresh comments for this post
      refreshComments(postId);
      fetchUserProfile();
    } catch (err) {
      setFeedbackNotice({
        type: "danger",
        msg: `Submission blocked: ${err.message}`
      });
      setTimeout(() => setFeedbackNotice(null), 5000);
    } finally {
      setSubmittingComment(prev => ({ ...prev, [postId]: false }));
    }
  }

  async function refreshComments(postId) {
    try {
      const freshComments = await api.getComments(postId);
      setPosts(posts.map(p => {
        if (p.id === postId) {
          return {
            ...p,
            comments: freshComments,
            comments_count: freshComments.length
          };
        }
        return p;
      }));
    } catch (e) {
      console.error("Failed to refresh comments:", e);
    }
  }

  async function handleCreatePost(e) {
    e.preventDefault();
    try {
      await api.createPost(newPost);
      setShowCreateModal(false);
      setNewPost({ title: "", content: "", post_type: "opinion", media_url: "" });
      fetchPosts();
      setFeedbackNotice({ type: "success", msg: "Post published to community feed!" });
      setTimeout(() => setFeedbackNotice(null), 4000);
    } catch (err) {
      alert(`Could not create post: ${err.message}`);
    }
  }

  async function handleAppealSubmit() {
    if (!appealModal || !appealReason.trim()) return;
    try {
      await api.submitAppeal(appealModal.comment_id, appealReason);
      setAppealModal(null);
      setAppealReason("");
      setFeedbackNotice({
        type: "success",
        msg: "Appeal submitted successfully! A human moderator will review it."
      });
      setTimeout(() => setFeedbackNotice(null), 5000);
    } catch (err) {
      alert(`Appeal failed: ${err.message}`);
    }
  }

  const isModeratorOrAdmin = user?.role === "moderator" || user?.role === "admin";
  const isInfluencer = user?.role === "influencer" || user?.role === "admin";

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
      
      {/* Restrictive Status Alert Banner if User is Rate Limited or Warned */}
      {userProfile && userProfile.active_restrictions && userProfile.active_restrictions.length > 0 && (
        <div className="bg-amber-50 border border-amber-300 rounded-2xl p-4 flex items-start space-x-3 text-amber-900 shadow-sm animate-pulse">
          <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
          <div className="text-xs">
            <p className="font-bold">Active Safety Policy Restrictions on your account:</p>
            <ul className="list-disc list-inside mt-1 space-y-0.5 text-amber-800">
              {userProfile.active_restrictions.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
            <p className="mt-1 text-[11px] text-amber-700">
              EchoSafe prioritizes restorative community health over permanent bans. Restrictions expire automatically.
            </p>
          </div>
        </div>
      )}

      {/* Floating Notice Toast */}
      {feedbackNotice && (
        <div className={`fixed top-20 right-6 max-w-sm p-4 rounded-xl shadow-xl border text-xs z-50 transition-all ${
          feedbackNotice.type === "success" ? "bg-emerald-50 border-emerald-300 text-emerald-900" :
          feedbackNotice.type === "warning" ? "bg-amber-50 border-amber-300 text-amber-900" :
          "bg-rose-50 border-rose-300 text-rose-900"
        }`}>
          <div className="flex items-center space-x-2">
            <span className="font-bold text-sm">
              {feedbackNotice.type === "success" ? "✓" : feedbackNotice.type === "warning" ? "⚠️" : "✕"}
            </span>
            <span className="font-medium">{feedbackNotice.msg}</span>
          </div>
        </div>
      )}

      {/* Header Bar: Filter tabs + Create Post Button */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
        <div className="flex items-center space-x-1 overflow-x-auto scrollbar-none py-1">
          {postTypeFilters.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveFilter(tab.id)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition ${
                activeFilter === tab.id
                  ? "bg-slate-900 text-white shadow-sm"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {isInfluencer && (
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center justify-center space-x-1.5 px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white text-xs font-bold rounded-xl shadow-sm transition"
          >
            <PlusCircle className="w-4 h-4" />
            <span>New Post</span>
          </button>
        )}
      </div>

      {/* Posts List */}
      {loading ? (
        <div className="text-center py-20 space-y-3">
          <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="text-xs text-slate-500 font-medium">Loading community feed...</p>
        </div>
      ) : posts.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-2xl border border-slate-200">
          <p className="text-sm font-semibold text-slate-700">No posts found in this category.</p>
          <p className="text-xs text-slate-400 mt-1">Try switching categories or submit a new post!</p>
        </div>
      ) : (
        <div className="space-y-6">
          {posts.map((post) => {
            const isCommentsOpen = expandedComments[post.id];
            const postComments = post.comments || [];

            return (
              <article key={post.id} className="bg-white rounded-2xl border border-slate-200/80 shadow-sm overflow-hidden hover:border-slate-300 transition duration-200">
                
                {/* Author Header */}
                <div className="p-4 flex items-center justify-between border-b border-slate-100">
                  <div className="flex items-center space-x-3">
                    <img
                      src={post.influencer_avatar || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"}
                      alt={post.influencer_name}
                      className="w-10 h-10 rounded-full object-cover border border-slate-200"
                    />
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-bold text-slate-900">{post.influencer_name}</span>
                        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200">
                          {post.influencer_niche || "Creator"}
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-400">
                        {post.created_at ? new Date(post.created_at).toLocaleDateString() : "Just now"}
                      </span>
                    </div>
                  </div>

                  <span className="text-[10px] uppercase font-bold tracking-wider px-2.5 py-1 rounded-lg bg-slate-100 text-slate-700">
                    {post.post_type}
                  </span>
                </div>

                {/* Post Content */}
                <div className="p-5 space-y-3">
                  <h3 className="font-bold text-base text-slate-900 leading-snug">{post.title}</h3>
                  <p className="text-xs text-slate-700 whitespace-pre-line leading-relaxed">{post.content}</p>

                  {/* Media visual preview if applicable */}
                  {post.media_url && (
                    <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-900 max-h-96 flex items-center justify-center">
                      <img
                        src={post.media_url}
                        alt={post.title}
                        className="w-full h-full object-cover max-h-96"
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    </div>
                  )}

                  {/* Specific styling for Opinion / Live style post */}
                  {post.post_type === "opinion" && (
                    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-l-4 border-blue-500 p-3 rounded-r-xl">
                      <p className="text-[11px] font-semibold text-blue-900">
                        💬 Community Debate Post: Keep discussion constructive and adhere to respectful engagement standards.
                      </p>
                    </div>
                  )}
                </div>

                {/* Action Bar (Likes, Comments toggle, Share) */}
                <div className="px-5 py-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-600 bg-slate-50/50">
                  <div className="flex items-center space-x-4">
                    <button
                      onClick={() => handleToggleLike(post.id)}
                      className={`flex items-center space-x-1.5 font-bold transition ${
                        post.has_liked ? "text-rose-600" : "hover:text-rose-600"
                      }`}
                    >
                      <Heart className={`w-4 h-4 ${post.has_liked ? "fill-rose-600 text-rose-600" : ""}`} />
                      <span>{post.likes_count || 0}</span>
                    </button>

                    <button
                      onClick={() => setExpandedComments(prev => ({ ...prev, [post.id]: !prev[post.id] }))}
                      className="flex items-center space-x-1.5 font-bold hover:text-blue-600 transition"
                    >
                      <MessageSquare className="w-4 h-4" />
                      <span>{post.comments_count || postComments.length || 0} Comments</span>
                    </button>
                  </div>

                  <span className="text-[11px] text-slate-400">EchoSafe AI Protected</span>
                </div>

                {/* Comments Section Drawer */}
                <div className="border-t border-slate-100 bg-slate-50 p-4 space-y-4">
                  
                  {/* Comment Input Box */}
                  <div className="flex items-start space-x-2">
                    <img
                      src={user?.avatar_url || "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"}
                      alt={user?.username}
                      className="w-8 h-8 rounded-full border border-slate-200 mt-0.5"
                    />
                    <div className="flex-1 flex items-center bg-white rounded-xl border border-slate-300 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-100 px-3 py-1.5 shadow-sm transition">
                      <input
                        type="text"
                        placeholder="Add a constructive comment..."
                        value={commentInputs[post.id] || ""}
                        onChange={(e) => setCommentInputs({ ...commentInputs, [post.id]: e.target.value })}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") handlePostComment(post.id);
                        }}
                        className="w-full text-xs text-slate-800 placeholder-slate-400 focus:outline-none"
                      />
                      <button
                        disabled={submittingComment[post.id] || !commentInputs[post.id]?.trim()}
                        onClick={() => handlePostComment(post.id)}
                        className="ml-2 text-blue-600 hover:text-blue-700 disabled:text-slate-300 transition"
                      >
                        <Send className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  {/* Comment Feed */}
                  <div className="space-y-2.5 pt-2">
                    {postComments.slice(0, isCommentsOpen ? undefined : 3).map((cmt) => {
                      const isHeld = cmt.status === "held_for_review";
                      const isAbusive = cmt.is_abusive || cmt.severity === "severe" || cmt.severity === "medium";
                      const shouldBlur = (cmt.is_blurred || isHeld || isAbusive) && !revealedAbuse[cmt.id];
                      const isAuthor = user?.username === cmt.username;

                      return (
                        <div
                          key={cmt.id}
                          className={`p-3 rounded-xl border transition text-xs ${
                            isHeld
                              ? "bg-amber-50/70 border-amber-200 text-amber-950"
                              : "bg-white border-slate-200/70 text-slate-800 shadow-2xs"
                          }`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center space-x-2">
                              <span className="font-bold text-slate-900">{cmt.username}</span>
                              {cmt.severity === "severe" && (
                                <span className="text-[9px] uppercase font-extrabold px-1.5 py-0.2 rounded bg-rose-100 text-rose-700">
                                  Flagged
                                </span>
                              )}
                              {isHeld && (
                                <span className="text-[9px] uppercase font-bold px-1.5 py-0.2 rounded bg-amber-200 text-amber-800">
                                  Under Review
                                </span>
                              )}
                            </div>
                            <span className="text-[10px] text-slate-400">
                              {cmt.created_at ? new Date(cmt.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : "Just now"}
                            </span>
                          </div>

                          {/* Content Display: Masked / Blurred with Click-To-Reveal for Safety */}
                          {shouldBlur ? (
                            <div className="space-y-1.5">
                              <p className="filter blur-xs select-none text-slate-400 font-mono text-[11px] bg-slate-100 px-2 py-1 rounded">
                                {cmt.masked_text || "[Comment masked by EchoSafe safety filter]"}
                              </p>
                              
                              <div className="flex items-center justify-between pt-1">
                                <span className="text-[10px] text-slate-500 italic flex items-center space-x-1">
                                  <ShieldAlert className="w-3 h-3 text-amber-600 inline mr-1" />
                                  <span>Hidden for safety (Hostility / policy trigger)</span>
                                </span>

                                {isModeratorOrAdmin ? (
                                  <button
                                    onClick={() => setRevealedAbuse(prev => ({ ...prev, [cmt.id]: true }))}
                                    className="text-[10px] font-bold text-indigo-600 hover:text-indigo-800 flex items-center space-x-1 px-2 py-0.5 rounded bg-indigo-50 border border-indigo-200"
                                  >
                                    <Eye className="w-3 h-3" />
                                    <span>Reveal (Moderator Only)</span>
                                  </button>
                                ) : (
                                  isAuthor && isHeld && (
                                    <button
                                      onClick={() => setAppealModal({ comment_id: cmt.id, text: cmt.masked_text })}
                                      className="text-[10px] font-bold text-amber-700 hover:text-amber-900 underline"
                                    >
                                      Submit Appeal
                                    </button>
                                  )
                                )}
                              </div>
                            </div>
                          ) : (
                            <div>
                              <p className="text-slate-800 leading-relaxed">{cmt.text}</p>
                              
                              {/* Explainability Tags for revealed comment */}
                              {isModeratorOrAdmin && cmt.explainability_keywords && cmt.explainability_keywords.length > 0 && (
                                <div className="mt-2 flex items-center space-x-1 text-[10px] text-slate-500">
                                  <span className="font-semibold text-rose-600">Trigger Words:</span>
                                  {cmt.explainability_keywords.map((kw, i) => (
                                    <span key={i} className="px-1.5 py-0.2 rounded bg-rose-50 text-rose-700 border border-rose-200 font-mono">
                                      {kw}
                                    </span>
                                  ))}
                                  <button
                                    onClick={() => setRevealedAbuse(prev => ({ ...prev, [cmt.id]: false }))}
                                    className="ml-auto text-[10px] text-slate-400 hover:text-slate-600 flex items-center space-x-1"
                                  >
                                    <EyeOff className="w-3 h-3" />
                                    <span>Re-hide</span>
                                  </button>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}

                    {postComments.length > 3 && !isCommentsOpen && (
                      <button
                        onClick={() => setExpandedComments(prev => ({ ...prev, [post.id]: true }))}
                        className="text-xs font-bold text-blue-600 hover:text-blue-700 pt-1"
                      >
                        View all {postComments.length} comments...
                      </button>
                    )}
                  </div>

                </div>

              </article>
            );
          })}
        </div>
      )}

      {/* Appeal Submission Modal */}
      {appealModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl border border-slate-200 space-y-4">
            <div className="flex items-center space-x-2 text-indigo-700">
              <HelpCircle className="w-5 h-5" />
              <h3 className="font-bold text-sm">Submit Fair Policy Appeal</h3>
            </div>
            
            <p className="text-xs text-slate-600">
              If you believe your comment was wrongly flagged as hostile or abusive, please describe your context below.
              Every appeal is reviewed by a human moderator.
            </p>

            <textarea
              rows={3}
              value={appealReason}
              onChange={(e) => setAppealReason(e.target.value)}
              placeholder="Explain why this comment adheres to respectful community standards..."
              className="w-full text-xs p-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-100 focus:border-indigo-500"
            />

            <div className="flex items-center justify-end space-x-2 pt-2">
              <button
                onClick={() => setAppealModal(null)}
                className="px-3.5 py-1.5 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100"
              >
                Cancel
              </button>
              <button
                onClick={handleAppealSubmit}
                disabled={!appealReason.trim()}
                className="px-4 py-1.5 rounded-xl text-xs font-bold bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
              >
                Submit Appeal
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Influencer Post Creation Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-slate-200 space-y-4">
            <h3 className="font-bold text-base text-slate-900">Publish New Creator Post</h3>
            
            <form onSubmit={handleCreatePost} className="space-y-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">Post Title</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. My Honest Take on Modern Tech Gadgets"
                  value={newPost.title}
                  onChange={(e) => setNewPost({ ...newPost, title: e.target.value })}
                  className="w-full text-xs p-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">Post Format</label>
                <select
                  value={newPost.post_type}
                  onChange={(e) => setNewPost({ ...newPost, post_type: e.target.value })}
                  className="w-full text-xs p-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-500"
                >
                  <option value="opinion">Opinion / Hot Take (Text Debate)</option>
                  <option value="reel">Reel (Short Video)</option>
                  <option value="image">Image Post</option>
                  <option value="video">Long-form Video</option>
                  <option value="live">Live Discussion Post</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">Content / Caption</label>
                <textarea
                  rows={4}
                  required
                  placeholder="Write your post caption or discussion question..."
                  value={newPost.content}
                  onChange={(e) => setNewPost({ ...newPost, content: e.target.value })}
                  className="w-full text-xs p-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-500"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">Media URL (Optional)</label>
                <input
                  type="url"
                  placeholder="https://images.unsplash.com/..."
                  value={newPost.media_url}
                  onChange={(e) => setNewPost({ ...newPost, media_url: e.target.value })}
                  className="w-full text-xs p-2.5 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-500"
                />
              </div>

              <div className="flex items-center justify-end space-x-2 pt-3">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl text-xs font-bold bg-blue-600 hover:bg-blue-700 text-white"
                >
                  Publish Post
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}