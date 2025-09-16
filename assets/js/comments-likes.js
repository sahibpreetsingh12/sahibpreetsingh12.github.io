class CommentsAndLikes {
    constructor(postSlug) {
        this.postSlug = postSlug;
        this.storageKeys = {
            comments: `comments_${postSlug}`,
            likes: `likes_${postSlug}`,
            dislikes: `dislikes_${postSlug}`,
            userLiked: `user_liked_${postSlug}`,
            userDisliked: `user_disliked_${postSlug}`
        };
        this.init();
    }

    init() {
        this.loadData();
        this.bindEvents();
        this.render();
    }

    loadData() {
        this.comments = JSON.parse(localStorage.getItem(this.storageKeys.comments)) || [];
        this.likes = parseInt(localStorage.getItem(this.storageKeys.likes)) || 0;
        this.dislikes = parseInt(localStorage.getItem(this.storageKeys.dislikes)) || 0;
        this.userLiked = localStorage.getItem(this.storageKeys.userLiked) === 'true';
        this.userDisliked = localStorage.getItem(this.storageKeys.userDisliked) === 'true';
    }

    saveData() {
        localStorage.setItem(this.storageKeys.comments, JSON.stringify(this.comments));
        localStorage.setItem(this.storageKeys.likes, this.likes.toString());
        localStorage.setItem(this.storageKeys.dislikes, this.dislikes.toString());
        localStorage.setItem(this.storageKeys.userLiked, this.userLiked.toString());
        localStorage.setItem(this.storageKeys.userDisliked, this.userDisliked.toString());
    }

    bindEvents() {
        document.addEventListener('click', (e) => {
            if (e.target.matches('#like-btn') || e.target.closest('#like-btn')) {
                this.handleLike();
            }
            if (e.target.matches('#dislike-btn') || e.target.closest('#dislike-btn')) {
                this.handleDislike();
            }
            if (e.target.matches('#submit-comment') || e.target.closest('#submit-comment')) {
                this.handleCommentSubmit();
            }
            if (e.target.matches('.delete-comment')) {
                this.handleCommentDelete(e.target.dataset.commentId);
            }
        });

        document.addEventListener('keypress', (e) => {
            if (e.target.matches('#comment-text') && e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                this.handleCommentSubmit();
            }
        });
    }

    handleLike() {
        if (this.userLiked) {
            this.likes--;
            this.userLiked = false;
        } else {
            this.likes++;
            this.userLiked = true;
            if (this.userDisliked) {
                this.dislikes--;
                this.userDisliked = false;
            }
        }
        this.saveData();
        this.updateLikeButtons();
    }

    handleDislike() {
        if (this.userDisliked) {
            this.dislikes--;
            this.userDisliked = false;
        } else {
            this.dislikes++;
            this.userDisliked = true;
            if (this.userLiked) {
                this.likes--;
                this.userLiked = false;
            }
        }
        this.saveData();
        this.updateLikeButtons();
    }

    handleCommentSubmit() {
        const nameInput = document.getElementById('comment-name');
        const textInput = document.getElementById('comment-text');

        const name = nameInput.value.trim();
        const text = textInput.value.trim();

        if (!name || !text) {
            alert('Please fill in both name and comment fields.');
            return;
        }

        const comment = {
            id: Date.now().toString(),
            name: name,
            text: text,
            timestamp: new Date().toISOString(),
            displayTime: new Date().toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            })
        };

        this.comments.unshift(comment);
        this.saveData();

        nameInput.value = '';
        textInput.value = '';

        this.renderComments();
    }

    handleCommentDelete(commentId) {
        if (confirm('Are you sure you want to delete this comment?')) {
            this.comments = this.comments.filter(comment => comment.id !== commentId);
            this.saveData();
            this.renderComments();
        }
    }

    updateLikeButtons() {
        const likeBtn = document.getElementById('like-btn');
        const dislikeBtn = document.getElementById('dislike-btn');
        const likeCount = document.getElementById('like-count');
        const dislikeCount = document.getElementById('dislike-count');

        if (likeBtn && likeCount) {
            likeBtn.classList.toggle('active', this.userLiked);
            likeCount.textContent = this.likes;
        }

        if (dislikeBtn && dislikeCount) {
            dislikeBtn.classList.toggle('active', this.userDisliked);
            dislikeCount.textContent = this.dislikes;
        }
    }

    renderComments() {
        const container = document.getElementById('comments-list');
        if (!container) return;

        if (this.comments.length === 0) {
            container.innerHTML = '<div class="no-comments">Be the first to comment!</div>';
            return;
        }

        const commentsHtml = this.comments.map(comment => `
            <div class="comment" data-comment-id="${comment.id}">
                <div class="comment-header">
                    <div class="comment-author">${this.escapeHtml(comment.name)}</div>
                    <div class="comment-meta">
                        <span class="comment-date">${comment.displayTime}</span>
                        <button class="delete-comment" data-comment-id="${comment.id}" title="Delete comment">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="comment-text">${this.escapeHtml(comment.text).replace(/\n/g, '<br>')}</div>
            </div>
        `).join('');

        container.innerHTML = commentsHtml;
    }

    render() {
        this.updateLikeButtons();
        this.renderComments();
    }

    escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
}

// Auto-initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Extract post slug from current page URL or use a fallback
    const pathSegments = window.location.pathname.split('/').filter(segment => segment);
    const postSlug = pathSegments[pathSegments.length - 1] || pathSegments[pathSegments.length - 2] || 'default';

    // Initialize the comments and likes system
    new CommentsAndLikes(postSlug);
});