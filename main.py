import os
import requests
import json
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional


class GNewsAICrawler:
    """
    GNews API Crawler for AI-related articles
    Enhanced with semantic search via OpenAI
    Deployed on Google Cloud Functions
    """

    BASE_URL = "https://gnews.io/api/v4/search"

    def __init__(self, api_key: str = None, openai_api_key: str = None):
        self.api_key = api_key or os.getenv('GNEWS_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("GNews API key required")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required")

    # ============================================
    # STEP 1: SEMANTIC QUERY GENERATION
    # ============================================
    # This is where natural language becomes keyword search.
    # Students type what they're looking for in plain English,
    # and GPT-4o-mini translates that into an optimized boolean
    # query that GNews can understand.
    #
    # WHY THIS MATTERS FOR TEACHING:
    # - Shows students that search engines don't "understand" meaning
    # - Makes the translation from intent → keywords visible
    # - Demonstrates how AI can bridge the gap between human
    #   language and machine search
    # ============================================

    def generate_search_query(self, natural_language_query: str, keyword: str) -> Dict:
        """
        Use OpenAI to translate a natural language query into an
        optimized GNews boolean search query.

        Args:
            natural_language_query: What the student typed in plain English
            keyword: The student's assigned AIReportage keyword

        Returns:
            Dict with generated query and the reasoning behind it
        """

        prompt = f"""You are a search query optimizer. A student is researching the topic 
of "{keyword}" in the context of artificial intelligence for an academic archive project.

The student described what they're looking for:
"{natural_language_query}"

Your job is to generate an optimized search query for the GNews API, which uses 
keyword-based boolean search (supports AND, OR, NOT, and parentheses).

Rules:
- Include relevant AI-related terms naturally (don't just append a generic AI block)
- Use synonyms and related terms the student might not have thought of
- Keep the query focused — don't make it so broad it returns noise
- Maximum ~120 characters for the query string
- Use boolean operators: AND, OR, NOT, parentheses for grouping
- NEVER use quotes (single or double) — the API does not support phrase matching
- Use only single words as search terms, not multi-word phrases

Respond in this exact JSON format (no markdown, no backticks):
{{
    "search_query": "your optimized boolean query here",
    "reasoning": "2-3 sentence explanation of why you chose these terms and structure",
    "terms_added": ["list", "of", "terms", "you", "added", "beyond", "what", "student", "said"],
    "terms_excluded": ["any", "terms", "you", "deliberately", "avoided"]
}}"""

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a precise search query optimizer. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Low temperature = more focused, less creative
                    "max_tokens": 300
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            # Parse the LLM's response
            content = data['choices'][0]['message']['content'].strip()
            # Clean potential markdown code fences
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            query_data = json.loads(content)

            # Strip any quotes the model may have included — GNews rejects them
            clean_query = query_data.get("search_query", "").replace("'", "").replace('"', "")

            return {
                "success": True,
                "search_query": clean_query,
                "reasoning": query_data.get("reasoning", ""),
                "terms_added": query_data.get("terms_added", []),
                "terms_excluded": query_data.get("terms_excluded", []),
                "original_input": natural_language_query
            }

        except Exception as e:
            print(f"OpenAI query generation failed: {e}")
            # Fallback: use the old-style static query
            return {
                "success": False,
                "search_query": self._fallback_query(keyword),
                "reasoning": "Fallback: OpenAI was unavailable, using standard keyword + AI terms.",
                "terms_added": [],
                "terms_excluded": [],
                "original_input": natural_language_query,
                "error": str(e)
            }

    def _fallback_query(self, keyword: str) -> str:
        """Original static query as fallback"""
        ai_terms = "(artificial intelligence OR AI OR machine learning OR neural OR deep learning)"
        return f"{keyword} AND {ai_terms}"

    # ============================================
    # STEP 2: FETCH ARTICLES FROM GNEWS
    # ============================================
    # This step is pure keyword search — GNews looks for the
    # exact terms in our query within article titles and content.
    # It has no understanding of meaning, just pattern matching.
    # ============================================

    def search_articles(self,
                        query: str,
                        keyword: str,
                        max_results: int = 30,
                        language: str = 'en',
                        search_in: str = 'title,content',
                        sort_by: str = 'relevance',
                        expand_content: bool = True) -> Dict:
        """
        Search for articles using GNews API with the generated query.

        Args:
            query: The optimized boolean search query (from Step 1)
            keyword: The student's assigned keyword (for metadata)
            max_results: Number of articles to fetch (fetch more than needed
                        so Step 3 can filter to the best ones)
            days_back: How far back to search (enterprise tier: up to years)
            language: Language code
            search_in: Where to search in articles
            sort_by: Sort order
            expand_content: Include full article content

        Returns:
            Dictionary with articles and metadata
        """

        # Fixed date range: June 1, 2023 to June 1, 2024
        start_date = datetime(2023, 6, 1, tzinfo=UTC)
        end_date = datetime(2024, 6, 1, tzinfo=UTC)

        params = {
            'apikey': self.api_key,
            'q': query,
            'lang': language,
            'max': min(max_results, 100),  # GNews caps at 100
            'from': start_date.isoformat(),
            'to': end_date.isoformat(),
            'in': search_in,
            'nullable': 'image',
            'sortby': sort_by
        }

        if expand_content:
            params['expand'] = 'content'

        try:
            print(f"[STEP 2] Searching GNews for: {query}")
            print(f"Date range: {start_date.date()} to {end_date.date()}")

            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            articles = self._process_articles(data.get('articles', []))

            result = {
                'success': True,
                'query_used': query,
                'keyword': keyword,
                'total_articles': data.get('totalArticles', 0),
                'articles': articles,
                'timestamp': datetime.now(UTC).isoformat(),
                'date_range': {
                    'from': start_date.isoformat(),
                    'to': end_date.isoformat()
                }
            }

            print(f"✓ Fetched {len(articles)} articles from GNews")
            return result

        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'query_used': query,
                'keyword': keyword,
                'timestamp': datetime.now(UTC).isoformat()
            }

    def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process and structure raw article data from GNews"""
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
            })

        return processed

    # ============================================
    # STEP 3: SEMANTIC RELEVANCE SCORING
    # ============================================
    # This is where the "understanding" happens. Unlike GNews
    # which just matched keywords, here the LLM actually reads
    # each article's title + description and judges how relevant
    # it is to what the student is actually looking for.
    #
    # WHY THIS MATTERS FOR TEACHING:
    # - Keyword search: "robot" matches an article about Robot
    #   vacuum cleaners — technically correct, semantically wrong
    # - Semantic scoring: the LLM understands that "robot" in the
    #   context of AI labor research means something different
    # - Students see the score AND the explanation, making the
    #   AI's reasoning transparent and debatable
    # ============================================

    def score_articles(self,
                       articles: List[Dict],
                       natural_language_query: str,
                       keyword: str,
                       top_n: int = 10) -> List[Dict]:
        """
        Use OpenAI to semantically score and rank articles.

        Args:
            articles: List of articles from GNews (Step 2 output)
            natural_language_query: The student's original query
            keyword: The assigned research keyword
            top_n: How many top articles to return

        Returns:
            List of articles, sorted by relevance, with scores and explanations
        """

        if not articles:
            return []

        # Build article representations for the LLM including content
        # Full content gives the model much better understanding of what
        # each article is actually about, beyond clickbait titles
        article_summaries = []
        for i, article in enumerate(articles):
            content_preview = (article.get('content') or '')[:500]
            article_summaries.append(
                f"[{i}] {article['title']}\n"
                f"    Source: {article['source_name']}\n"
                f"    Description: {article['description'][:200]}\n"
                f"    Content: {content_preview}"
            )

        articles_text = "\n\n".join(article_summaries)

        prompt = f"""You are evaluating search results for an academic AI research archive.

RESEARCH KEYWORD: "{keyword}"
SEARCH INTENT: "{natural_language_query}"

Below are {len(articles)} articles returned by a keyword search. For each article, 
assign a relevance score from 0-100 and a brief explanation.

Scoring criteria:
- 80-100: Directly addresses the search intent AND relates to AI
- 60-79: Related to the topic but tangential, or only loosely connected to AI
- 40-59: Mentions relevant terms but is really about something else
- 0-39: Not relevant (e.g., keyword coincidence, different domain entirely)

Important: In your explanations, do NOT reference "the student" or "student's intent". 
Just explain what the article is about and why it scored the way it did.

ARTICLES:
{articles_text}

Respond with ONLY a JSON array (no markdown, no backticks). Each element should be:
{{
    "index": 0,
    "score": 85,
    "explanation": "Brief 1-sentence reason for this score"
}}"""

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a precise academic relevance evaluator. Always respond with valid JSON arrays only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,  # Very low = consistent scoring
                    "max_tokens": 2000
                },
                timeout=45
            )
            response.raise_for_status()
            data = response.json()

            content = data['choices'][0]['message']['content'].strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            scores = json.loads(content)

            # Merge scores back into articles
            score_map = {s['index']: s for s in scores}

            for i, article in enumerate(articles):
                if i in score_map:
                    article['relevance_score'] = score_map[i]['score']
                    article['score_explanation'] = score_map[i]['explanation']
                else:
                    article['relevance_score'] = 50  # Default if scoring missed it
                    article['score_explanation'] = "No score generated"

            # Sort by relevance score (highest first) and take top N
            articles.sort(key=lambda a: a.get('relevance_score', 0), reverse=True)
            top_articles = articles[:top_n]

            # Re-rank after sorting
            for i, article in enumerate(top_articles, 1):
                article['semantic_rank'] = i

            print(f"✓ Scored {len(articles)} articles, returning top {len(top_articles)}")
            return top_articles

        except Exception as e:
            print(f"OpenAI scoring failed: {e}")
            # Fallback: return articles as-is with no scores
            for article in articles[:top_n]:
                article['relevance_score'] = None
                article['score_explanation'] = f"Scoring unavailable: {str(e)}"
                article['semantic_rank'] = article['rank']
            return articles[:top_n]

    # ============================================
    # STEP 4: KEYWORD CONTEXT ANALYSIS
    # ============================================
    # For each article, the LLM reads the content and explains
    # how the student's assigned keyword is used or implied in
    # the article, and provides a contextual definition.
    #
    # This is central to AIReportage's pedagogical goal:
    # students don't just find articles that mention a keyword —
    # they develop an interpretive understanding of how that
    # concept operates in different contexts.
    #
    # Example: "Robot" in a healthcare article might mean
    # "an AI-assisted surgical system that extends a surgeon's
    # precision" — very different from "robot" in a manufacturing
    # article where it means "an autonomous machine replacing
    # human labor on an assembly line."
    # ============================================

    VALID_KEYWORDS = [
        'authorship', 'data', 'embodiment', 'hallucination',
        'imagination', 'innovation', 'labor', 'neural',
        'policy', 'risk', 'robot', 'sentience', 'trust', 'visualization'
    ]

    def analyze_keyword_context(self, articles: List[Dict], keyword: str) -> List[Dict]:
        """
        Use OpenAI to analyze how the assigned keyword is used or
        implied in each article's content.

        Args:
            articles: List of scored articles (Step 3 output)
            keyword: The student's assigned AIReportage keyword

        Returns:
            Same articles list with 'keyword_usage' and 'keyword_definition' added
        """

        if not articles:
            return articles

        # Validate keyword
        keyword_lower = keyword.strip().lower()
        if keyword_lower not in self.VALID_KEYWORDS:
            print(f"Warning: '{keyword}' not in standard keyword list, proceeding anyway")

        # Build article representations
        article_texts = []
        for i, article in enumerate(articles):
            content = (article.get('content') or '')[:600]
            article_texts.append(
                f"[{i}] {article['title']}\n"
                f"    Content: {content}"
            )

        articles_block = "\n\n".join(article_texts)

        prompt = f"""You are an academic research assistant for AIReportage.org, an archive 
studying how AI concepts appear in news media.

THE KEYWORD: "{keyword}"

Below are {len(articles)} articles. For each article, provide a one-sentence 
definition of what "{keyword}" means within the specific context of that article. 

This should NOT be a dictionary definition — it should capture how this particular 
article frames, uses, or implies the concept of "{keyword}". If the keyword doesn't 
appear explicitly, define it based on the contextual connection between the article's 
content and the concept.

ARTICLES:
{articles_block}

Respond with ONLY a JSON array (no markdown, no backticks). Each element:
{{
    "index": 0,
    "keyword_definition": "one sentence contextual definition"
}}"""

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a precise academic research assistant. Always respond with valid JSON arrays only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                timeout=45
            )
            response.raise_for_status()
            data = response.json()

            content = data['choices'][0]['message']['content'].strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            analyses = json.loads(content)

            # Merge into articles
            analysis_map = {a['index']: a for a in analyses}

            for i, article in enumerate(articles):
                if i in analysis_map:
                    article['keyword_definition'] = analysis_map[i].get('keyword_definition', '')
                else:
                    article['keyword_definition'] = ''

            print(f"✓ Analyzed keyword context for {len(articles)} articles")
            return articles

        except Exception as e:
            print(f"OpenAI keyword analysis failed: {e}")
            for article in articles:
                article['keyword_definition'] = f"Analysis unavailable: {str(e)}"
            return articles


# ============================================
# GOOGLE CLOUD FUNCTION ENTRY POINT
# ============================================
# This orchestrates the 4-step pipeline:
#   1. Natural language → keyword query (OpenAI)
#   2. Keyword query → articles (GNews)
#   3. Articles → scored & ranked articles (OpenAI)
#   4. Articles → keyword context analysis (OpenAI)
# ============================================

def crawl_articles(request):
    """
    Main entry point for Google Cloud Function

    Expected JSON body:
    {
        "keyword": "robot",
        "natural_query": "How are robots replacing warehouse workers?",
        "max_results": 10,
        "use_semantic": true
    }

    Date range is fixed to June 1, 2023 – June 1, 2024.

    When use_semantic is true (default), the full 3-step pipeline runs.
    When false, it falls back to the original keyword-only search.
    This toggle lets students compare the two approaches side by side.

    Returns: JSON with articles, query metadata, and scoring transparency
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

    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
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
        natural_query = request_json.get('natural_query', '')
        max_results = request_json.get('max_results', 10)
        use_semantic = request_json.get('use_semantic', True)
        fetch_count = request_json.get('fetch_count', 30)  # Fetch more, return fewer

        # Initialize crawler
        crawler = GNewsAICrawler()

        # ---- STEP 1: Generate optimized query ----
        if use_semantic and natural_query:
            print("[PIPELINE] Step 1: Generating semantic search query...")
            query_result = crawler.generate_search_query(natural_query, keyword)
            search_query = query_result['search_query']
        else:
            # Classic mode: static keyword + AI terms
            search_query = crawler._fallback_query(keyword)
            query_result = {
                'success': True,
                'search_query': search_query,
                'reasoning': 'Classic mode: using standard keyword + AI boolean terms.',
                'terms_added': [],
                'terms_excluded': [],
                'original_input': natural_query or keyword
            }

        # ---- STEP 2: Fetch articles from GNews ----
        print("[PIPELINE] Step 2: Fetching articles from GNews...")
        gnews_result = crawler.search_articles(
            query=search_query,
            keyword=keyword,
            max_results=fetch_count if use_semantic else max_results
        )

        if not gnews_result['success']:
            return (json.dumps(gnews_result), 500, headers)

        # ---- STEP 3: Score and rank articles ----
        if use_semantic and natural_query and gnews_result['articles']:
            print("[PIPELINE] Step 3: Scoring articles for semantic relevance...")
            scored_articles = crawler.score_articles(
                articles=gnews_result['articles'],
                natural_language_query=natural_query,
                keyword=keyword,
                top_n=max_results
            )
        else:
            scored_articles = gnews_result['articles'][:max_results]

        # ---- STEP 4: Keyword context analysis ----
        if scored_articles:
            print("[PIPELINE] Step 4: Analyzing keyword context in articles...")
            scored_articles = crawler.analyze_keyword_context(
                articles=scored_articles,
                keyword=keyword
            )

        # ---- BUILD RESPONSE ----
        # Include all the transparency data students need to see
        result = {
            'success': True,
            'mode': 'semantic' if (use_semantic and natural_query) else 'classic',

            # Step 1 transparency
            'query_generation': {
                'student_input': natural_query or keyword,
                'generated_query': query_result['search_query'],
                'reasoning': query_result['reasoning'],
                'terms_added': query_result.get('terms_added', []),
                'terms_excluded': query_result.get('terms_excluded', [])
            },

            # Step 2 metadata
            'gnews_metadata': {
                'total_available': gnews_result.get('total_articles', 0),
                'fetched': len(gnews_result.get('articles', [])),
                'returned': len(scored_articles),
                'date_range': gnews_result.get('date_range', {})
            },

            # Step 3 results
            'articles': scored_articles,

            'keyword': keyword,
            'timestamp': datetime.now(UTC).isoformat()
        }

        return (json.dumps(result), 200, headers)

    except Exception as e:
        return (
            json.dumps({
                'success': False,
                'error': f'Internal error: {str(e)}'
            }),
            500,
            headers
        )
        