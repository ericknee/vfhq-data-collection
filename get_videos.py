import yt_dlp
import os
import json
import re

CACHE_FILE = 'video_cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as file:
        json.dump(cache, file, indent=4)

def convert_filename(filename):
    return re.sub(r'[\/:*?"<>|]', '_', filename)

def download_4k_videos(celebrity_names, download_path='videos', max_videos_per_celebrity=20):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    video_cache = load_cache()

    ydl_opts = {
        'format': 'bestvideo[height=2160]+bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
    }

    for name in celebrity_names:
        if name not in video_cache:
            video_cache[name] = []

        query = f"{name} interview 4k"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                results = ydl.extract_info(f'ytsearch{max_videos_per_celebrity}:{query}', download=False)['entries']
                download_count = 0

                for video in results:
                    
                    if download_count >= max_videos_per_celebrity:
                        break

                    video_url = video['webpage_url']
                    if video_url in video_cache[name]:
                        print(f"Skipping {video_url}, already in cache")
                        continue
                    save_cache(video_cache)
                    try:
                        # Check if both video and audio streams are available
                        info_dict = ydl.extract_info(video_url, download=False)
                        formats = info_dict.get('formats', [])
                        has_4k_video = any(f.get('height') == 2160 and f.get('vcodec') != 'none' for f in formats)
                        has_audio = any(f.get('acodec') != 'none' for f in formats)
                        # Exclude shorts
                        duration = info_dict.get('duration')
                        if has_4k_video and has_audio and duration and duration >= 60: # [1, 25] minutes
                            # Dynamically choose the best format available
                            format_string = 'bestvideo[height=2160]+bestaudio/best'
                            ydl.download([f'{video_url}+{format_string}'])
                            video_cache[name].append(video_url)
                            save_cache(video_cache)  # Save cache after each download
                            download_count += 1
                        else:
                            print(f"Skipping {video_url}, does not meet criteria (4K video, audio, and longer than 60 seconds)")
                    
                    except Exception as e:
                        print(f"Error downloading {video_url}: {e}")

            except Exception as e:
                print(f"Error searching for {name}: {e}")

    

# 200 popular celebrities
celebrities = ["Dwayne 'The Rock' Johnson", "Will Smith", "Ellen DeGeneres", "Jimmy Fallon", "Oprah Winfrey", "James Corden", "Kevin Hart", "Ryan Reynolds", "Kim Kardashian", "Kanye West", "Justin Bieber", "Ariana Grande", "Selena Gomez", "Taylor Swift", "Billie Eilish", "Cardi B", "Kendall Jenner", "Kylie Jenner", "David Dobrik", "Lilly Singh", "John Krasinski", "Chris Hemsworth", "Jennifer Lawrence", "Emma Stone", "Zendaya", "Tom Holland", "Chris Evans", "Scarlett Johansson", "Robert Downey Jr.", "Mark Ruffalo", "Brie Larson", "Gal Gadot", "Henry Cavill", "Jason Momoa", "Ben Affleck", "Margot Robbie", "Joaquin Phoenix", "Brad Pitt", "Leonardo DiCaprio", "Angelina Jolie", "Matt Damon", "Anne Hathaway", "Reese Witherspoon", "Sandra Bullock", "Meryl Streep", "Julia Roberts", "Denzel Washington", "Morgan Freeman", "Tom Cruise", "Hugh Jackman", "Jake Gyllenhaal", "Ryan Gosling", "Emma Watson", "Daniel Radcliffe", "RuPaul", "Trevor Noah", "Stephen Colbert", "Seth Meyers", "Conan O'Brien", "David Letterman", "Jay Leno", "Jimmy Kimmel", "Graham Norton", "Trevor Noah", "James Charles", "Jeffree Star", "Shane Dawson", "PewDiePie", "Markiplier", "Jacksepticeye", "Ninja", "MrBeast", "Logan Paul", "Jake Paul", "Liza Koshy", "Lele Pons", "Jojo Siwa", "Miranda Sings", "Casey Neistat", "Philip DeFranco", "Rhett and Link", "PewDiePie", "Marques Brownlee", "Unbox Therapy", "Linus Tech Tips", "TechLinked", "Austin Evans", "Jonathan Morrison", "Dave Lee", "iJustine", "Tyler Oakley", "Shay Carl", "Roman Atwood", "Casey Neistat", "H3H3 Productions", "Lilly Singh", "Superwoman", "Nash Grier", "Cameron Dallas", "Madison Beer", "Dixie D'Amelio", "Charli D'Amelio", "Addison Rae", "Tana Mongeau", "Gabbie Hanna", "NikkieTutorials", "Michelle Phan", "Bethany Mota", "Rosanna Pansino", "Jenn Im", "Smosh", "Fine Brothers", "Epic Meal Time", "Philip DeFranco", "Liza Koshy", "David Dobrik", "Jenna Marbles", "Shane Dawson", "Jackie Aina", "Desi Perkins", "PatrickStarrr", "Manny MUA", "James Charles", "Jaclyn Hill", "Jeffree Star", "Huda Kattan", "Wayne Goss", "Lisa Eldridge", "Pixiwoo", "Michelle Phan", "Tati Westbrook", "Kathleen Lights", "Emily Noel", "NikkieTutorials", "Chloe Morello", "Zoella", "Tanya Burr", "Ingrid Nilsen", "Bethany Mota", "MyLifeAsEva", "Rachel Levin", "Alisha Marie", "Aspyn Ovard", "Niki and Gabi", "Sierra Furtado", "Meredith Foster", "Manny MUA", "PatrickStarrr", "Kathleen Lights", "Desi Perkins", "Jackie Aina", "James Charles", "Jaclyn Hill", "Jeffree Star", "Tati Westbrook", "RawBeautyKristi", "Thataylaa", "NikkieTutorials", "Chloe Morello", "Zoella", "Tanya Burr", "Ingrid Nilsen", "Bethany Mota", "MyLifeAsEva", "Rachel Levin", "Alisha Marie", "Aspyn Ovard", "Niki and Gabi", "Sierra Furtado", "Meredith Foster", "Superwoman", "Tyler Oakley", "Grace Helbig", "Mamrie Hart", "Hannah Hart", "Kingsley", "Connor Franta", "Ricky Dillon", "Kian Lawley", "JC Caylen", "Trevor Moran", "Andrea Russett", "Jenn McAllister", "Meghan Rienks", "Megan DeAngelis", "Alex Wassabi", "Aaron Burriss", "Gabriel Conte", "Jess Conte", "The Dolan Twins", "Ethan Dolan", "Grayson Dolan", "Colleen Ballinger", "Miranda Sings", "Todrick Hall", "PewDiePie", "Markiplier", "Jacksepticeye", "Ninja", "MrBeast", "Logan Paul", "Jake Paul", "Liza Koshy", "Lele Pons", "Jojo Siwa"]



download_4k_videos(celebrities)
