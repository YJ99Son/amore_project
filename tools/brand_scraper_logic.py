import requests
from bs4 import BeautifulSoup
from collections import Counter
from io import BytesIO
from PIL import Image
import json
import os

# ==========================================
# 1. Configuration & Target URLs
# ==========================================
BRAND_URLS = {
    "Sulwhasoo": "https://www.sulwhasoo.com/kr/ko/about/brand-story.html",
    "Laneige": "https://www.laneige.com/kr/ko/brand/index.html",
    "Innisfree": "https://www.innisfree.com/kr/ko/brand/story.do",
    # Add more brands here
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}

# ==========================================
# 2. Text Extraction Logic (Story & Tone)
# ==========================================
def extract_brand_text(url):
    """
    Logic:
    1. Visit the Brand Story page.
    2. Extract text from <p> tags and specific class names common in brand sites.
    3. Summarize or extract keywords (simple heuristic here).
    """
    print(f"[*] Scraping Text from: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Heuristic: Brand stories are usually in <p> tags with substantial length
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        
        # Filter short UI texts (nav bars, footers)
        story_text = [p for p in paragraphs if len(p) > 50]
        
        # Join top 3 paragraphs as the main story
        full_story = " ".join(story_text[:3])
        
        return full_story
    except Exception as e:
        print(f"[!] Error scraping text: {e}")
        return ""

# ==========================================
# 3. Visual Extraction Logic (Colors)
# ==========================================
def extract_dominant_colors(image_url=None, num_colors=3):
    """
    Logic:
    1. Download the Hero Image or maintain a Screenshot of the page.
    2. Use K-Means clustering or ColorThief to find dominant colors.
    3. Return Hex codes.
    
    *Note: For this demo, we simulate image analysis with a placeholder logic 
     unless a real image URL is provided.
    """
    print(f"[*] Analyzing Colors for visual elements...")
    
    if not image_url:
        # In a real agentic workflow, we would use a headless browser to take a screenshot
        # screenshot_path = browser.take_screenshot()
        # image = Image.open(screenshot_path)
        return ["#Placeholder_Color1", "#Placeholder_Color2", "#Placeholder_Color3"]
    
    try:
        # Actual Logic if we had an image
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Resize for speed
        img = img.resize((150, 150))
        result = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        
        # Extract palette
        palette = result.getpalette()
        color_counts = sorted(result.getcolors(), reverse=True)
        colors = []
        
        for i in range(num_colors):
            palette_index = color_counts[i][1]
            dominant_color = palette[palette_index*3:palette_index*3+3]
            hex_color = '#%02x%02x%02x' % tuple(dominant_color)
            colors.append(hex_color)
            
        return colors
        
    except Exception as e:
        print(f"[!] Error analyzing colors: {e}")
        return []

# ==========================================
# 4. Main Execution Loop
# ==========================================
def main():
    knowledge_base = {}
    
    for brand, url in BRAND_URLS.items():
        print(f"\nProcessing Brand: {brand}")
        
        # 1. Text Logic
        story = extract_brand_text(url)
        
        # 2. Visual Logic (Simulated)
        # In production, we would find the <meta property="og:image"> tag
        colors = extract_dominant_colors() 
        
        knowledge_base[brand] = {
            "story": story[:200] + "...", # Truncate for display
            "visual_colors": colors,
            "url": url
        }
        
    # Save Output
    with open('brand_extraction_result.json', 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        
    print("\n[V] Extraction Complete. Saved to brand_extraction_result.json")

if __name__ == "__main__":
    # Note: Requires 'requests', 'beautifulsoup4', 'pillow'
    # pip install requests beautifulsoup4 pillow
    main()
