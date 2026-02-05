import os
import requests
import json
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional

class GNewsAICrawler:
    """
    GNews API Crawler for AI-related articles
    Deployed on Google Cloud Functions
    """
    
    BASE_URL = "https://gnews.io/api/v4/search"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GNEWS_API_KEY')
        if not self.api_key:
            raise ValueError("GNews API key required")
    
    def search_articles(self, 
                       keyword: str,
                       max_results: int = 10,
                       days_back: int = 30,  # Changed for free tier
                       language: str = 'en',
                       search_in: str = 'title,content',
                       sort_by: str = 'relevance',
                       expand_content: bool = False) -> Dict:  # Changed default to False for free tier
        """
        Search for articles using GNews API
        
        Args:
            keyword: Primary search keyword (e.g., 'robot', 'hallucination')
            max_results: Number of articles to return (1-100)
            days_back: How many days back to search (max 30 for free tier)
            language: Language code ('en' for English)
            search_in: Where to search ('title,content', 'title,description', etc.)
            sort_by: Sort order ('relevance' or 'publishedAt')
            expand_content: Include full article content (limited on free tier)
            
        Returns:
            Dictionary with articles and metadata
        """
        
        # Build AI-focused query
        ai_terms = "(artificial intelligence OR AI OR machine learning OR neural OR deep learning)"
        query = f"{keyword} AND {ai_terms}"
        
        # Calculate date range for free tier (last 30 days max)
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=min(days_back, 30))
        
        # Build request parameters
        params = {
            'apikey': self.api_key,
            'q': query,
            'lang': language,
            'max': max_results,
            'from': start_date.isoformat(),
            'to': end_date.isoformat(),
            'in': search_in,
            'nullable': 'image',
            'sortby': sort_by
        }
        
        if expand_content:
            params['expand'] = 'content'
        
        try:
            print(f"Searching GNews for: {query}")
            print(f"Date range: {start_date.date()} to {end_date.date()}")
            
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            result = {
                'success': True,
                'query': query,
                'keyword': keyword,
                'total_articles': data.get('totalArticles', 0),
                'articles': self._process_articles(data.get('articles', [])),
                'timestamp': datetime.now(UTC).isoformat(),
                'date_range': {
                    'from': start_date.isoformat(),
                    'to': end_date.isoformat()
                }
            }
            
            print(f"✓ Found {len(result['articles'])} articles")
            return result
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'keyword': keyword,
                'timestamp': datetime.now(UTC).isoformat()
            }
    
    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process and structure article data"""
        processed = []
        
        for idx, article in enumerate(articles, 1):
            processed.append({
                'rank': idx,
                'title': article.get('title', 'No title'),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'url': article.get('url', ''),
                'image': article.get('image', ''),
                'published_at': article.get('publishedAt', ''),
                'source_name': article.get('source', {}).get('name', 'Unknown'),
                'source_url': article.get('source', {}).get('url', ''),
                # Placeholders for bonus features
                'ai_summary': None,
                'relevance_score': None,
                'keep_recommendation': None
            })
        
        return processed


# ============================================
# GOOGLE CLOUD FUNCTION ENTRY POINT
# ============================================
def crawl_articles(request):
    """
    Main entry point for Google Cloud Function
    
    Expected JSON body:
    {
        "keyword": "robot",
        "max_results": 10,
        "days_back": 30
    }
    
    Returns: JSON with articles or error
    """
    
    # Handle CORS for Apps Script
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    try:
        request_json = request.get_json(silent=True)
        
        if not request_json or 'keyword' not in request_json:
            return (
                json.dumps({
                    'success': False,
                    'error': 'Missing required parameter: keyword'
                }),
                400,
                headers
            )
        
        keyword = request_json['keyword']
        max_results = request_json.get('max_results', 10)
        days_back = request_json.get('days_back', 30)
        
        # Initialize crawler
        crawler = GNewsAICrawler()
        
        # Search articles
        result = crawler.search_articles(
            keyword=keyword,
            max_results=max_results,
            days_back=days_back
        )
        
        status_code = 200 if result['success'] else 500
        
        return (json.dumps(result), status_code, headers)
        
    except Exception as e:
        return (
            json.dumps({
                'success': False,
                'error': f'Internal error: {str(e)}'
            }),
            500,
            headers
        )