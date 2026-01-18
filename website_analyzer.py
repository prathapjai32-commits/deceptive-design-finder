"""
Website Analyzer - Extract dark pattern features from websites
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time
from screenshot_capture import PatternScreenshotCapture

class WebsiteAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Keywords for detecting dark patterns
        self.urgency_keywords = [
            'limited time', 'only today', 'ends soon', 'hurry', 'act now', 'last chance',
            'expires', 'limited offer', 'only left', 'few remaining', 'don\'t miss',
            'urgent', 'time sensitive', 'exclusive', 'once in a lifetime', 'flash sale',
            'deal expires', 'closing soon', 'final hours', 'almost gone', 'selling fast'
        ]
        
        self.confirm_shaming_keywords = [
            'no thanks', 'i\'m not interested', 'maybe later', 'skip this deal',
            'no, i don\'t want to save', 'decline', 'pass', 'not now', 'maybe next time'
        ]
        
        self.hidden_cost_keywords = [
            'processing fee', 'service charge', 'convenience fee', 'handling fee',
            'additional charges', 'plus shipping', 'taxes extra', 'hidden fee'
        ]
        
        self.subscription_keywords = [
            'auto-renew', 'recurring', 'monthly charge', 'annual subscription',
            'cancel anytime', 'free trial', 'subscription', 'membership fee'
        ]
    
    def fetch_website(self, url, timeout=10):
        """Fetch website content"""
        try:
            # Ensure URL has scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            response = requests.get(url, headers=self.headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text, response.url
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching website: {str(e)}")
    
    def analyze_html(self, html_content):
        """Analyze HTML content for dark patterns"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True).lower()
        
        # Get all HTML content (for tag analysis)
        html_str = str(soup).lower()
        
        features = {}
        
        # 1. Urgency Language
        urgency_count = sum(text.count(keyword) for keyword in self.urgency_keywords)
        features['urgency_language'] = min(urgency_count, 10.0)
        
        # 2. Hidden Costs
        hidden_cost_count = sum(text.count(keyword) for keyword in self.hidden_cost_keywords)
        features['hidden_costs'] = min(hidden_cost_count, 5.0)
        
        # 3. Misleading Labels (check for deceptive button text)
        misleading_patterns = [
            r'continue.*free', r'claim.*now', r'get.*free.*now',
            r'click.*here.*free', r'no.*charge.*now'
        ]
        misleading_count = sum(len(re.findall(pattern, text)) for pattern in misleading_patterns)
        features['misleading_labels'] = min(misleading_count, 8.0)
        
        # 4. Forced Continuity (subscription indicators)
        subscription_count = sum(text.count(keyword) for keyword in self.subscription_keywords)
        # Check for cancel/cancel subscription mentions (might indicate difficulty)
        cancel_mentions = text.count('cancel')
        features['forced_continuity'] = min(subscription_count * 0.5 + (cancel_mentions > 3) * 3, 6.0)
        
        # 5. Roach Motel (easy signup, hard cancellation)
        signup_easy = len(soup.find_all(['button', 'a'], string=re.compile(r'sign up|register|join|subscribe', re.I)))
        cancel_hard = text.count('cancel subscription') + text.count('how to cancel')
        features['roach_motel'] = min((signup_easy > 5) * 3 + (cancel_hard < 2 and subscription_count > 2) * 4, 7.0)
        
        # 6. Trick Questions (complex opt-out, confusing checkboxes)
        checkbox_count = len(soup.find_all('input', {'type': 'checkbox'}))
        opt_in_default = len(soup.find_all('input', {'checked': True, 'type': 'checkbox'}))
        features['trick_questions'] = min((opt_in_default / max(checkbox_count, 1)) * 5, 5.0) if checkbox_count > 0 else 0
        
        # 7. Sneak into Basket (pre-selected addons, extra items)
        addon_keywords = ['add to cart', 'included', 'recommended for you', 'frequently bought together']
        addon_count = sum(text.count(keyword) for keyword in addon_keywords)
        preselected = len(soup.find_all('input', {'checked': True, 'type': 'checkbox', 'name': re.compile(r'addon|extra|item', re.I)}))
        features['sneak_into_basket'] = min(addon_count * 0.5 + preselected * 2, 4.0)
        
        # 8. Confirm Shaming
        shaming_count = sum(text.count(keyword) for keyword in self.confirm_shaming_keywords)
        features['confirm_shaming'] = min(shaming_count, 6.0)
        
        # 9. Disguised Ads (check for sponsored content markers)
        ad_indicators = ['sponsored', 'promoted', 'advertisement', 'affiliate link']
        ad_count = sum(html_str.count(indicator) for indicator in ad_indicators)
        # Check for links with rel="sponsored" or "nofollow" in suspicious contexts
        sponsored_links = len(soup.find_all('a', rel=re.compile(r'sponsored|nofollow')))
        features['disguised_ads'] = min((ad_count + sponsored_links * 0.5), 5.0)
        
        # 10. Price Comparison Prevention
        price_obfuscation = text.count('price') > 10 and text.count('compare') < 2
        no_price_display = soup.find_all(string=re.compile(r'\$\d+\.\d+', re.I))
        features['price_comparison_prevention'] = min((price_obfuscation * 2 + len(no_price_display) < 5 * 2), 4.0)
        
        # 11. Popup Frequency (check for popup triggers)
        popup_triggers = ['popup', 'modal', 'overlay', 'lightbox']
        popup_count = sum(html_str.count(trigger) for trigger in popup_triggers)
        # Check for onbeforeunload, exit intent scripts
        exit_intent = text.count('exit') + text.count('beforeunload')
        features['popup_frequency'] = min((popup_count + exit_intent) * 0.5, 10.0)
        
        # 12. Opt-Out Difficulty
        opt_out_links = soup.find_all(['a', 'button'], string=re.compile(r'opt.?out|unsubscribe|remove', re.I))
        # If subscriptions exist but opt-out is hard to find
        opt_out_difficulty = (subscription_count > 2) and (len(opt_out_links) < 2)
        features['opt_out_difficulty'] = min(opt_out_difficulty * 6 + (len(opt_out_links) == 0 and subscription_count > 0) * 2, 8.0)
        
        # 13. Fake Reviews
        review_count = text.count('review') + text.count('rating')
        # Check for suspicious review patterns (all 5 stars, no negative)
        star_patterns = len(re.findall(r'5\s*(?:star|★)', text))
        suspicious_reviews = (star_patterns > 10) and (text.count('1 star') + text.count('2 star') < 3)
        features['fake_reviews'] = min((suspicious_reviews * 5 + (review_count > 20) * 2), 7.0)
        
        # 14. Social Proof Manipulation
        social_proof_keywords = ['people also bought', 'customers who viewed', 'trending now',
                                 'bought together', 'limited stock', 'only 3 left in stock',
                                 'other people viewing', 'in your area']
        social_proof_count = sum(text.count(keyword) for keyword in social_proof_keywords)
        # Check for fake scarcity indicators
        fake_scarcity = len(re.findall(r'only \d+ left|just \d+ remaining|less than \d+', text))
        features['social_proof_manipulation'] = min((social_proof_count + fake_scarcity) * 0.5, 6.0)
        
        return features, soup
    
    def analyze_url(self, url):
        """Analyze a website URL for dark patterns"""
        try:
            # Fetch website
            html_content, final_url = self.fetch_website(url)
            
            # Analyze content
            features, soup = self.analyze_html(html_content)
            
            # Get page metadata
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "No title"
            
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ''
            
            # Capture visual evidence of dark patterns
            screenshot_capture = PatternScreenshotCapture()
            evidence = {
                'fake_reviews': screenshot_capture.capture_pattern_evidence(html_content, soup, final_url, 'fake_reviews'),
                'sneak_into_basket': screenshot_capture.capture_pattern_evidence(html_content, soup, final_url, 'sneak_into_basket'),
                'confirm_shaming': screenshot_capture.capture_pattern_evidence(html_content, soup, final_url, 'confirm_shaming')
            }
            
            return {
                'features': features,
                'url': final_url,
                'title': title_text,
                'description': description[:200] if description else '',
                'evidence': evidence,
                'soup': soup,
                'html_content': html_content,
                'success': True
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'features': None
            }
