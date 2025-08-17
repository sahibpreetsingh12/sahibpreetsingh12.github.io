# Git Workflow for Your Personal Website

## 🚀 When You Want to Add New Features

### 1. **Start with Clean Main Branch**
```bash
# Always start from main branch
git checkout main

# Pull latest changes
git pull origin main

# Check status (should be clean)
git status
```

### 2. **Create Feature Branch**
```bash
# Create and switch to new feature branch
git checkout -b feature/your-feature-name

# Examples:
git checkout -b feature/new-blog-post
git checkout -b feature/portfolio-updates
git checkout -b feature/performance-improvements
```

### 3. **Work on Your Changes**
```bash
# Make your changes to files
# Edit, add, modify as needed

# Check what you've changed
git status
git diff

# Stage specific files
git add filename.html
git add folder/

# Or stage everything
git add .
```

### 4. **Commit Your Changes**
```bash
# Commit with descriptive message
git commit -m "feat: add new blog post about AI projects

- Added new post about recent AI work
- Updated navigation to include projects section
- Fixed mobile responsiveness issue

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 5. **Push and Deploy**
```bash
# Push feature branch first (for backup)
git push -u origin feature/your-feature-name

# Switch to main and merge
git checkout main
git merge feature/your-feature-name

# Push to deploy live
git push origin main
```

## 🔧 Quick Commands for Small Changes

### **For Simple Updates (blog posts, content changes):**
```bash
# Work directly on main for small changes
git checkout main
git pull origin main

# Make your changes
# ...

git add .
git commit -m "content: update about section with recent achievements"
git push origin main
```

## 🛠️ Useful Git Commands

### **Check Status**
```bash
git status              # See what's changed
git log --oneline -5    # See recent commits
git diff                # See exact changes
```

### **Undo Changes**
```bash
git checkout filename   # Undo changes to specific file
git reset --hard        # Undo all uncommitted changes (careful!)
```

### **Branch Management**
```bash
git branch              # List branches
git branch -d branch-name  # Delete branch after merging
git checkout main       # Switch to main branch
```

## 🚨 Important Rules

1. **Always pull before starting work**: `git pull origin main`
2. **Use descriptive commit messages**: Explain what and why
3. **Test before pushing to main**: Your live site updates automatically
4. **Keep commits small and focused**: One feature per commit when possible
5. **Use feature branches for big changes**: Don't break the live site

## 🎯 Commit Message Types

- `feat:` - New features
- `fix:` - Bug fixes  
- `content:` - Content updates (blog posts, text changes)
- `style:` - CSS/design changes
- `chore:` - Maintenance tasks
- `docs:` - Documentation updates

## 🌐 Deployment Notes

- **GitHub Pages auto-deploys** when you push to main
- **Changes go live in 2-3 minutes** after push
- **Test locally first** before pushing to main
- **System monitor and chatbot** need to be running locally for full functionality

## 🆘 If Something Goes Wrong

```bash
# See what happened
git log --oneline -10

# Go back to previous commit (replace COMMIT_HASH)
git reset --hard COMMIT_HASH
git push --force origin main  # Only if necessary!

# Or create a revert commit (safer)
git revert HEAD
git push origin main
```

Remember: Your website automatically deploys from the main branch, so always test your changes before pushing to main!