import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from bs4 import BeautifulSoup
import os
import torch
import time
import random
from urllib.parse import urljoin, urlparse
import json
import re
from io import BytesIO
import psycopg2
from g4f.client import Client
import warnings
warnings.filterwarnings('ignore')

class AdvancedFaceMatchingCrawler:
    def __init__(self, match_threshold=0.75, quality_threshold=50):
        """
        Initialize the advanced face matching crawler
        
        Args:
            match_threshold: Cosine similarity threshold for face matching (0-1)
            quality_threshold: Minimum quality score for detected faces
        """
        self.match_threshold = match_threshold
        self.quality_threshold = quality_threshold
        
        # Check GPU availability
        print(f"🔧 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔧 GPU name: {torch.cuda.get_device_name(0)}")
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            print("🔧 Using CPU")
        
        # Load CLIP model for embeddings
        print("📦 Loading CLIP model...")
        self.model = SentenceTransformer('clip-ViT-B-32', device=self.device)
        print("✅ CLIP model loaded successfully!")
        
        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Setup session for web requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        # Results storage
        self.results = {
            'input_faces': [],
            'crawled_websites': [],
            'matches_found': [],
            'summaries': {}
        }
    
    def detect_faces_in_image(self, image_path):
        """
        Detect and extract faces from an input image
        """
        print(f"🔍 Detecting faces in: {image_path}")
        
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return []
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Improve contrast
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            print(f"📸 Found {len(faces)} faces in input image")
            
            detected_faces = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Add padding around face
                padding = max(int(w * 0.1), int(h * 0.1))
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(img.shape[1] - x_pad, w + 2 * padding)
                h_pad = min(img.shape[0] - y_pad, h + 2 * padding)
                
                # Extract face region
                face_crop = img[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
                
                # Convert to RGB for PIL
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                
                # Check face quality
                quality_score = self.assess_face_quality(face_crop)
                
                if quality_score >= self.quality_threshold:
                    # Create embedding
                    embedding = self.model.encode(face_pil)
                    
                    face_info = {
                        'face_id': i,
                        'coordinates': (x, y, w, h),
                        'quality_score': quality_score,
                        'embedding': embedding,
                        'face_image': face_pil
                    }
                    
                    detected_faces.append(face_info)
                    print(f"✅ Face {i+1}: Quality {quality_score:.1f} - Added for matching")
                else:
                    print(f"❌ Face {i+1}: Quality {quality_score:.1f} - Skipped (too low)")
            
            self.results['input_faces'] = detected_faces
            return detected_faces
            
        except Exception as e:
            print(f"❌ Error detecting faces: {e}")
            return []
    
    def assess_face_quality(self, face_crop):
        """Assess the quality of a detected face"""
        try:
            if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                return 0
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Check for eyes
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3)
            has_eyes = len(eyes) >= 1
            
            # Calculate quality metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Quality scoring
            quality_score = 0
            
            # Size bonus
            area = face_crop.shape[0] * face_crop.shape[1]
            quality_score += min(25, area / 1000)
            
            # Eye detection bonus
            if has_eyes:
                quality_score += 30
            
            # Brightness (50-200 is good)
            if 50 <= brightness <= 200:
                quality_score += 25
            else:
                quality_score += max(0, 25 - abs(brightness - 125) / 5)
            
            # Contrast
            if contrast > 20:
                quality_score += 20
            else:
                quality_score += max(0, contrast)
            
            return min(100, quality_score)
            
        except Exception:
            return 0
    
    def extract_images_from_webpage(self, url):
        """Extract all image URLs from a webpage"""
        try:
            print(f"🌐 Crawling webpage: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            image_urls = []
            
            # Find all image tags
            img_tags = soup.find_all('img')
            
            for img in img_tags:
                img_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                
                if img_url:
                    # Convert relative URLs to absolute
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    elif img_url.startswith('/'):
                        img_url = urljoin(url, img_url)
                    elif not img_url.startswith('http'):
                        img_url = urljoin(url, img_url)
                    
                    # Filter potential face images
                    if self.is_potential_face_image(img_url, img):
                        image_urls.append(img_url)
            
            print(f"📷 Found {len(image_urls)} potential images to analyze")
            return image_urls
            
        except Exception as e:
            print(f"❌ Error crawling {url}: {e}")
            return []
    
    def is_potential_face_image(self, img_url, img_tag):
        """Filter images that might contain faces"""
        # Skip obvious non-face images
        skip_keywords = ['logo', 'icon', 'banner', 'background', 'button', 'arrow']
        url_lower = img_url.lower()
        
        for keyword in skip_keywords:
            if keyword in url_lower:
                return False
        
        # Prefer images with face-related keywords
        face_keywords = ['photo', 'profile', 'headshot', 'portrait', 'person', 'face']
        alt_text = (img_tag.get('alt') or '').lower()
        
        for keyword in face_keywords:
            if keyword in alt_text:
                return True
        
        return True
    
    def download_and_process_image(self, img_url, website_url):
        """Download image and detect faces, create embeddings"""
        try:
            # Add delay to be respectful
            time.sleep(random.uniform(0.2, 0.5))
            
            response = self.session.get(img_url, timeout=8)
            if response.status_code != 200:
                return None
            
            # Load image
            img_bytes = np.frombuffer(response.content, np.uint8)
            cv_image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                return None
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return {
                    'img_url': img_url,
                    'faces_found': 0,
                    'face_embeddings': [],
                    'matches': []
                }
            
            face_embeddings = []
            matches = []
            
            # Process each detected face
            for face_idx, (x, y, w, h) in enumerate(faces):
                # Extract face
                face_crop = cv_image[y:y+h, x:x+w]
                
                # Assess quality
                quality_score = self.assess_face_quality(face_crop)
                
                if quality_score >= self.quality_threshold:
                    # Convert to RGB for PIL
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    
                    # Create embedding
                    web_embedding = self.model.encode(face_pil)
                    
                    face_info = {
                        'face_idx': face_idx,
                        'coordinates': (x, y, w, h),
                        'quality_score': quality_score,
                        'embedding': web_embedding
                    }
                    
                    face_embeddings.append(face_info)
                    
                    # Compare with input faces
                    for input_face in self.results['input_faces']:
                        similarity = self.calculate_cosine_similarity(
                            input_face['embedding'], 
                            web_embedding
                        )
                        
                        if similarity >= self.match_threshold:
                            match_info = {
                                'input_face_id': input_face['face_id'],
                                'web_face_idx': face_idx,
                                'similarity': similarity,
                                'quality_score': quality_score,
                                'img_url': img_url,
                                'website_url': website_url,
                                'coordinates': (x, y, w, h)
                            }
                            
                            matches.append(match_info)
                            print(f"🎯 MATCH FOUND! Similarity: {similarity:.3f} on {website_url}")
            
            return {
                'img_url': img_url,
                'faces_found': len(faces),
                'quality_faces': len(face_embeddings),
                'face_embeddings': face_embeddings,
                'matches': matches
            }
            
        except Exception as e:
            print(f"❌ Error processing {img_url}: {e}")
            return None
    
    def calculate_cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    def scrape_webpage_content(self, url):
        """Scrape content from webpage for summary generation"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to get main content
            content_selectors = [
                'main', 'article', '[role="main"]',
                '.content', '.main-content', '.page-content',
                'div.content', 'div.main', 'div.article'
            ]
            
            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'span'])
            else:
                text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
            
            # Extract meaningful text
            content_parts = []
            for element in text_elements[:50]:  # Limit to avoid too much text
                text = element.get_text(strip=True)
                if len(text) > 20:  # Only meaningful content
                    content_parts.append(text)
            
            content = ' '.join(content_parts)
            return content[:3000]  # Limit content length
            
        except Exception as e:
            print(f"❌ Error scraping content from {url}: {e}")
            return f"Could not scrape content from {url}"
    
    def generate_gpt4o_summary(self, website_url, content, match_info):
        """Generate summary using GPT-4o"""
        try:
            client = Client()
            
            prompt = f"""Based on the following webpage content, provide a professional summary about the person whose face was matched in an image.

Website: {website_url}
Match Quality: {match_info['similarity']:.3f} similarity, {match_info['quality_score']:.1f} quality score

Content:
{content}

Please provide:
1. A professional summary of the person (2-3 sentences)
2. Key details: position, organization, achievements
3. Relevant professional information

Keep it concise and professional. Focus on career, education, and accomplishments."""

            print(f"🤖 Generating GPT-4o summary for {website_url}")
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                web_search=True
            )
            
            summary = response.choices[0].message.content.strip()
            print(f"✅ Summary generated successfully")
            return summary
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def crawl_and_match(self, input_image_path, website_urls):
        """Main function to crawl websites and find face matches"""
        print(f"🚀 Starting Advanced Face Matching Crawler")
        print(f"📸 Input image: {input_image_path}")
        print(f"🌐 Websites to crawl: {len(website_urls)}")
        print(f"🎯 Match threshold: {self.match_threshold}")
        print("=" * 60)
        
        # Step 1: Detect faces in input image
        input_faces = self.detect_faces_in_image(input_image_path)
        
        if not input_faces:
            print("❌ No quality faces found in input image. Exiting.")
            return
        
        print(f"✅ Ready to match against {len(input_faces)} input face(s)")
        
        # Step 2: Process each website
        all_matches = []
        
        for website_idx, url in enumerate(website_urls, 1):
            print(f"\n🌐 Processing website {website_idx}/{len(website_urls)}: {url}")
            
            try:
                # Extract images from webpage
                image_urls = self.extract_images_from_webpage(url)
                
                if not image_urls:
                    print("❌ No images found on this webpage")
                    continue
                
                # Process each image
                website_matches = []
                processed_images = 0
                
                for img_idx, img_url in enumerate(image_urls, 1):
                    print(f"  📷 Processing image {img_idx}/{len(image_urls)}: {img_url[:60]}...")
                    
                    result = self.download_and_process_image(img_url, url)
                    
                    if result:
                        processed_images += 1
                        if result['matches']:
                            website_matches.extend(result['matches'])
                            all_matches.extend(result['matches'])
                
                # Store website results
                website_result = {
                    'url': url,
                    'images_found': len(image_urls),
                    'images_processed': processed_images,
                    'matches_found': len(website_matches)
                }
                
                self.results['crawled_websites'].append(website_result)
                
                print(f"📊 Website summary: {len(image_urls)} images found, {processed_images} processed, {len(website_matches)} matches")
                
                # Generate summary if matches found
                if website_matches:
                    print(f"🤖 Generating summary for matches on {url}")
                    
                    content = self.scrape_webpage_content(url)
                    best_match = max(website_matches, key=lambda x: x['similarity'])
                    summary = self.generate_gpt4o_summary(url, content, best_match)
                    
                    self.results['summaries'][url] = {
                        'summary': summary,
                        'matches_count': len(website_matches),
                        'best_similarity': best_match['similarity'],
                        'content_scraped': len(content)
                    }
                else:
                    print("❌ No matches found on this website")
                
            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
                continue
        
        # Step 3: Generate final report
        self.results['matches_found'] = all_matches
        self.generate_final_report()
        
        return self.results
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        matches = self.results['matches_found']
        websites = self.results['crawled_websites']
        summaries = self.results['summaries']
        
        print(f"\n" + "=" * 60)
        print("🎯 ADVANCED FACE MATCHING RESULTS")
        print("=" * 60)
        
        print(f"📸 Input faces processed: {len(self.results['input_faces'])}")
        print(f"🌐 Websites crawled: {len(websites)}")
        print(f"📷 Total images processed: {sum(w['images_processed'] for w in websites)}")
        print(f"🎯 Total matches found: {len(matches)}")
        print(f"📝 Summaries generated: {len(summaries)}")
        
        if matches:
            print(f"\n📋 DETAILED MATCHES:")
            
            for i, match in enumerate(matches, 1):
                print(f"\n🔍 Match {i}:")
                print(f"   Website: {match['website_url']}")
                print(f"   Image: {match['img_url'][:80]}...")
                print(f"   Similarity: {match['similarity']:.3f}")
                print(f"   Quality: {match['quality_score']:.1f}")
                print(f"   Input Face ID: {match['input_face_id']}")
        
        if summaries:
            print(f"\n📝 GENERATED SUMMARIES:")
            
            for url, summary_data in summaries.items():
                print(f"\n🌐 {url}")
                print(f"   Matches: {summary_data['matches_count']}")
                print(f"   Best Similarity: {summary_data['best_similarity']:.3f}")
                print(f"   Summary: {summary_data['summary'][:200]}...")
        
        # Save detailed report
        report_file = 'advanced_face_matching_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self.results.copy()
            json_results['input_faces'] = [
                {k: v.tolist() if isinstance(v, np.ndarray) else v 
                 for k, v in face.items() if k != 'face_image'}
                for face in self.results['input_faces']
            ]
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        return self.results

def main():
    """Example usage"""
    
    # Initialize crawler
    crawler = AdvancedFaceMatchingCrawler(
        match_threshold=0.75,  # 75% similarity threshold
        quality_threshold=50   # Minimum quality score
    )
    
    # Input image path
    input_image = r"D:\SAKIB\NSU\COURSES\CSE 499B\FIRST BENCHMARK\CODE\FACIAL RECOGNITION\test_image\ec066937-9d83-4b05-a471-f8d19f8acf7e.jpg"
    
    # Websites to crawl
    websites = [
        "https://www.i4cp.com/leading-the-way/leading-the-way-podcast-shark-tanks-mark-cuban",
        "https://www.scmp.com/magazines/style/celebrity/article/3189475/inside-mark-cubans-humble-parenting-style-shark-tank",
        "https://www.britannica.com/money/Mark-Cuban",
        "https://markcubancompanies.com/marks-bio/",
        "https://www.i4cp.com/leading-the-way/leading-the-way-podcast-shark-tanks-mark-cuban",
        "https://www.caa.com/caaspeakers/gary-vaynerchuk",
        "https://www.apbspeakers.com/speaker/gary-vaynerchuk/",
        "https://cew.org/people/gary-vaynerchuk/",
        "https://www.adma.com.au/people/gary-vaynerchuk",
        "https://www.castmagic.io/creators/gary-vaynerchuk",
        "https://www.businessinsider.com/gary-vaynerchuk-ai-chatgpt-chatbot-microsoft-google-job-labor-trends-2023-2",
        "https://rocketdevs.com/blog/top-angel-investors-list",
        "https://www.crunchbase.com/person/naval-ravikant",
        "https://twit.tv/people/kevin-rose",
        "https://www.cryptotimes.io/2024/03/11/moonbirds-creator-kevin-rose-sells-1-2-million-nfts/",
        "https://techcrunch.com/2017/04/03/kevin-rose-is-going-back-to-cali-and-joining-true-ventures-as-venture-partner/",
        "https://www.celebritynetworth.com/richest-businessmen/ceos/kevin-rose-net-worth/",
        "https://successsolver.com/profiles/tim-ferriss-net-worth/",
        "https://www.growth-hackers.net/what-is-tim-ferriss-net-worth-how-make-money-investor-rich-successful-entrepreneur-business/",
        "https://blogs.the-hospitalist.org/content/listening-tim-ferriss",
        "https://techcrunch.com/2018/09/05/cyan-banister-shares-her-journey-from-homeless-teen-to-vc/",
        "https://techcrunch.com/2020/03/02/cyan-banister-leaves-founders-fund-for-long-journey-ventures/",
        "https://techcrunch.com/2019/11/15/vc-cyan-banister-on-her-path/"
    ]
    
    # Run the crawler
    results = crawler.crawl_and_match(input_image, websites)
    
    print(f"\n🎉 Crawling completed!")
    print(f"📊 Found {len(results['matches_found'])} total matches")
    
    return results

if __name__ == "__main__":
    main()