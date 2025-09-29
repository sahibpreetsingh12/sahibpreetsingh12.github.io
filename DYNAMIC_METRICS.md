# Dynamic Status Metrics

## Overview
The "Current Status" section now displays **real-time, always-changing metrics** that prove continuous progress and activity. Every visitor sees updated numbers.

## What's Dynamic Now

### 1. **GitHub Contributions** (This Year)
- Pulls from GitHub Events API
- Counts: PushEvents, PullRequests, Issues, CreateEvents
- Updates every page load
- Shows THIS YEAR's activity (resets Jan 1)

### 2. **Blog Posts Written**
- Counts posts from Jekyll's `site.posts`
- Grows with every new blog post you publish
- Automatically updates when you deploy

### 3. **Days Building**
- Calculated from your GitHub account creation date
- Grows by 1 every day (automatically)
- Shows long-term commitment

### 4. **Current Streak**
- Calculates consecutive days with GitHub activity
- Based on recent 300 events
- Resets if you miss a day (motivation!)

## How It Works

### Data Sources
1. **GitHub Events API**: `https://api.github.com/users/sahibpreetsingh12/events?per_page=300`
2. **GitHub User API**: `https://api.github.com/users/sahibpreetsingh12`
3. **Jekyll site.posts**: Blog post count

### Caching Strategy
- Data cached in localStorage
- Falls back to cached data if API unavailable
- Updates on every page load when online
- Smooth animations when numbers change

### Performance
- Async fetching (non-blocking)
- Rate limit aware (uses cached data)
- Minimal API calls (300 events covers ~30 days)

## Why This Works

### Shows Real Progress
- Every commit = contributions ↑
- Every blog post = posts ↑
- Every day = days building ↑
- Every active day = streak ↑

### Always Positive
- Numbers only go UP
- Proves continuous work
- Visitors see you're actively shipping

### Social Proof
- Real data from GitHub
- Can't be faked
- Third-party verified

## Future Enhancements

Possible additions:
- Total repos count
- Stars received on projects
- Lines of code written (WakaTime integration)
- Projects deployed count
- Kaggle competition rank

## Testing

Open browser console to see logs:
```
✅ Status metrics updated: 247 contributions, 12 day streak
✅ Days coding: 1847
✅ Blog post count: 5
```

## Deployment

When you push to GitHub Pages:
1. GitHub contributions update automatically
2. Blog count updates from _posts folder
3. Days building increases daily
4. Streak shows current activity

No manual updates needed - it's all automatic! 🚀