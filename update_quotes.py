#!/usr/bin/env python3
"""
Script to update README.md with a daily motivational quote.
Rotates through a curated list of motivational quotes.
"""

import os
import re
from datetime import datetime

# Curated list of motivational quotes
QUOTES = [
    {
        "quote": "The secret of getting ahead is getting started.",
        "author": "Mark Twain"
    },
    {
        "quote": "Success is not final, failure is not fatal: it is the courage to continue that counts.",
        "author": "Winston Churchill"
    },
    {
        "quote": "Don't watch the clock; do what it does. Keep going.",
        "author": "Sam Levenson"
    },
    {
        "quote": "The future belongs to those who believe in the beauty of their dreams.",
        "author": "Eleanor Roosevelt"
    },
    {
        "quote": "Believe you can and you're halfway there.",
        "author": "Theodore Roosevelt"
    },
    {
        "quote": "It does not matter how slowly you go as long as you do not stop.",
        "author": "Confucius"
    },
    {
        "quote": "Everything you've ever wanted is on the other side of fear.",
        "author": "George Addair"
    },
    {
        "quote": "Success is not how high you have climbed, but how you make a positive difference to the world.",
        "author": "Roy T. Bennett"
    },
    {
        "quote": "The only way to do great work is to love what you do.",
        "author": "Steve Jobs"
    },
    {
        "quote": "Your limitation—it's only your imagination.",
        "author": "Unknown"
    },
    {
        "quote": "Push yourself, because no one else is going to do it for you.",
        "author": "Unknown"
    },
    {
        "quote": "Great things never come from comfort zones.",
        "author": "Unknown"
    },
    {
        "quote": "Dream it. Wish it. Do it.",
        "author": "Unknown"
    },
    {
        "quote": "Success doesn't just find you. You have to go out and get it.",
        "author": "Unknown"
    },
    {
        "quote": "The harder you work for something, the greater you'll feel when you achieve it.",
        "author": "Unknown"
    },
    {
        "quote": "Dream bigger. Do bigger.",
        "author": "Unknown"
    },
    {
        "quote": "Don't stop when you're tired. Stop when you're done.",
        "author": "Unknown"
    },
    {
        "quote": "Wake up with determination. Go to bed with satisfaction.",
        "author": "Unknown"
    },
    {
        "quote": "Do something today that your future self will thank you for.",
        "author": "Sean Patrick Flanery"
    },
    {
        "quote": "Little things make big days.",
        "author": "Unknown"
    },
    {
        "quote": "It's going to be hard, but hard does not mean impossible.",
        "author": "Unknown"
    },
    {
        "quote": "Don't wait for opportunity. Create it.",
        "author": "Unknown"
    },
    {
        "quote": "Sometimes we're tested not to show our weaknesses, but to discover our strengths.",
        "author": "Unknown"
    },
    {
        "quote": "The key to success is to focus on goals, not obstacles.",
        "author": "Unknown"
    },
    {
        "quote": "Dream it. Believe it. Build it.",
        "author": "Unknown"
    },
    {
        "quote": "What you get by achieving your goals is not as important as what you become by achieving your goals.",
        "author": "Zig Ziglar"
    },
    {
        "quote": "The only limit to our realization of tomorrow will be our doubts of today.",
        "author": "Franklin D. Roosevelt"
    },
    {
        "quote": "You are never too old to set another goal or to dream a new dream.",
        "author": "C.S. Lewis"
    },
    {
        "quote": "Act as if what you do makes a difference. It does.",
        "author": "William James"
    },
    {
        "quote": "The way to get started is to quit talking and begin doing.",
        "author": "Walt Disney"
    }
]


def get_daily_quote():
    """
    Get a quote based on the day of the year to ensure daily rotation.
    """
    day_of_year = datetime.now().timetuple().tm_yday
    quote_index = day_of_year % len(QUOTES)
    return QUOTES[quote_index]


def update_readme():
    """
    Update README.md with a new daily motivational quote.
    """
    readme_path = "README.md"
    
    if not os.path.exists(readme_path):
        print(f"Error: {readme_path} not found!")
        return False
    
    # Read current README content
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get today's quote
    quote_data = get_daily_quote()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Create the new motivation section
    new_motivation_section = f"""## 💪 Daily Motivation

Get inspired with a new motivational quote every day! This section is automatically updated daily at midnight UTC.

> **Quote of the Day:**
> 
> *"{quote_data['quote']}"*
> 
> — {quote_data['author']}

---

*Last updated: {today} | Automatically updated via GitHub Actions*"""
    
    # Pattern to match the motivation section
    pattern = r'## 💪 Daily Motivation\n\n.*?\n\n---\n\n\*Last updated:.*?\*'
    
    # Check if the motivation section exists
    if re.search(pattern, content, re.DOTALL):
        # Replace existing motivation section
        updated_content = re.sub(pattern, new_motivation_section, content, flags=re.DOTALL)
    else:
        # Add motivation section at the end
        updated_content = content.rstrip() + "\n\n---\n\n" + new_motivation_section + "\n"
    
    # Write updated content back to README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"✅ README updated successfully with quote by {quote_data['author']}")
    print(f"📅 Date: {today}")
    print(f"💬 Quote: {quote_data['quote']}")
    return True


if __name__ == "__main__":
    update_readme()
