import yt_dlp
import os
import json
import re
import subprocess

CACHE_FILE = 'video_cache.json'

def convert_filename(filename):
    return re.sub(r'[\/:*?"<>|]', '_', filename)

def update_status(json_file, parameter_name, parameter_value):
    status = load_cache()
    if parameter_name in status:
        status[parameter_name] += parameter_value
    else:
        status[parameter_name] = parameter_value
    save_cache(status)
    
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            data = json.load(file)
            # Convert lists back to sets
            return {key: set(value) for key, value in data.items()}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as file:
        # Convert sets to lists for JSON serialization
        data = {key: list(value) for key, value in cache.items()}
        json.dump(data, file, indent=4)

def get_downloaded_videos(download_path):
    downloaded_videos = set()
    for root, dirs, files in os.walk(download_path):
        for file in files:
            if file.endswith('.mp4'):
                title = os.path.splitext(file)[0]
                downloaded_videos.add(title)
    return downloaded_videos

def download_4k_videos(celebrity_names, download_path='videos', max_videos_per_celebrity=20):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    video_cache = load_cache()
    downloaded_videos = get_downloaded_videos(download_path)

    ydl_opts = {
        'format': 'bestvideo[height=2160]+bestaudio/best',
        'noplaylist': True,
        'quiet': True,
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(download_path, '%(title)s.%(ext)s'),
    }

    for name in celebrity_names:
        if name not in video_cache:
            video_cache[name] = set()

        query = f"{name} interview 4k"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                results = ydl.extract_info(f'ytsearch{max_videos_per_celebrity}:{query}', download=False)['entries']
                download_count = 0

                for video in results:
                    
                    if download_count >= max_videos_per_celebrity:
                        break

                    video_url = video['webpage_url']
                    video_title = video['title']

                    if video_url in video_cache[name]:
                        print(f"Skipping {video_url}, already in cache")
                        continue

                    if video_title in downloaded_videos:
                        print(f"Video {video_title} already downloaded, adding to cache")
                        video_cache[name].add(video_url)
                        save_cache(video_cache)
                        continue

                    try:
                        video_cache[name].add(video_url)
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
                            # video_cache[name].add(video_url)
                            download_count += 1
                        else:
                            print(f"Skipping {video_url}, does not meet criteria (4K video, audio, and longer than 60 seconds)")
                        save_cache(video_cache)
                    except Exception as e:
                        print(f"Error downloading {video_url}: {e}")

            except Exception as e:
                print(f"Error searching for {name}: {e}")


def extract_frames(video_path, output_base_folder="processed-videos", desired_fps=6):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_folder = os.path.join(output_base_folder, video_name)
    print(f"Extracting frames from {video_name}")
    output_folder = os.path.join(video_folder, 'frames')
    original_fps = get_video_fps(video_path)
    if os.path.isdir(output_folder) and os.listdir(output_folder):
        extracted_frame_count = len(os.listdir(output_folder))
        video_duration = get_video_duration(video_path)
        extracted_fps = extracted_frame_count / video_duration
        print(f"FRAMES ALREADY EXTRACTED FOR {video_name}")
        return video_folder, extracted_fps, original_fps, video_duration
    
    os.makedirs(output_folder, exist_ok=True)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={desired_fps}',
        os.path.join(output_folder, 'frame_%05d.jpg')
    ]

    subprocess.run(command, check=True)
    
    extracted_frame_count = len(os.listdir(output_folder))
    video_duration = get_video_duration(video_path)
    extracted_fps = extracted_frame_count / video_duration
    update_status('stats.json', 'total_videos_processed', 1)
    return video_folder, extracted_fps, original_fps, video_duration


# 200 popular celebrities
celebrities = ["Dwayne 'The Rock' Johnson", "Will Smith", "Ellen DeGeneres", "Jimmy Fallon", "Oprah Winfrey", "James Corden", "Kevin Hart", "Ryan Reynolds", "Kim Kardashian", "Kanye West", "Justin Bieber", "Ariana Grande", "Selena Gomez", "Taylor Swift", "Billie Eilish", "Cardi B", "Kendall Jenner", "Kylie Jenner", "David Dobrik", "Lilly Singh", "John Krasinski", "Chris Hemsworth", "Jennifer Lawrence", "Emma Stone", "Zendaya", "Tom Holland", "Chris Evans", "Scarlett Johansson", "Robert Downey Jr.", "Mark Ruffalo", "Brie Larson", "Gal Gadot", "Henry Cavill", "Jason Momoa", "Ben Affleck", "Margot Robbie", "Joaquin Phoenix", "Brad Pitt", "Leonardo DiCaprio", "Angelina Jolie", "Matt Damon", "Anne Hathaway", "Reese Witherspoon", "Sandra Bullock", "Meryl Streep", "Julia Roberts", "Denzel Washington", "Morgan Freeman", "Tom Cruise", "Hugh Jackman", "Jake Gyllenhaal", "Ryan Gosling", "Emma Watson", "Daniel Radcliffe", "RuPaul", "Trevor Noah", "Stephen Colbert", "Seth Meyers", "Conan O'Brien", "David Letterman", "Jay Leno", "Jimmy Kimmel", "Graham Norton", "Trevor Noah", "James Charles", "Jeffree Star", "Shane Dawson", "PewDiePie", "Markiplier", "Jacksepticeye", "Ninja", "MrBeast", "Logan Paul", "Jake Paul", "Liza Koshy", "Lele Pons", "Jojo Siwa", "Miranda Sings", "Casey Neistat", "Philip DeFranco", "Rhett and Link", "PewDiePie", "Marques Brownlee", "Unbox Therapy", "Linus Tech Tips", "TechLinked", "Austin Evans", "Jonathan Morrison", "Dave Lee", "iJustine", "Tyler Oakley", "Shay Carl", "Roman Atwood", "Casey Neistat", "H3H3 Productions", "Lilly Singh", "Superwoman", "Nash Grier", "Cameron Dallas", "Madison Beer", "Dixie D'Amelio", "Charli D'Amelio", "Addison Rae", "Tana Mongeau", "Gabbie Hanna", "NikkieTutorials", "Michelle Phan", "Bethany Mota", "Rosanna Pansino", "Jenn Im", "Smosh", "Fine Brothers", "Epic Meal Time", "Philip DeFranco", "Liza Koshy", "David Dobrik", "Jenna Marbles", "Shane Dawson", "Jackie Aina", "Desi Perkins", "PatrickStarrr", "Manny MUA", "James Charles", "Jaclyn Hill", "Jeffree Star", "Huda Kattan", "Wayne Goss", "Lisa Eldridge", "Pixiwoo", "Michelle Phan", "Tati Westbrook", "Kathleen Lights", "Emily Noel", "NikkieTutorials", "Chloe Morello", "Zoella", "Tanya Burr", "Ingrid Nilsen", "Bethany Mota", "MyLifeAsEva", "Rachel Levin", "Alisha Marie", "Aspyn Ovard", "Niki and Gabi", "Sierra Furtado", "Meredith Foster", "Manny MUA", "PatrickStarrr", "Kathleen Lights", "Desi Perkins", "Jackie Aina", "James Charles", "Jaclyn Hill", "Jeffree Star", "Tati Westbrook", "RawBeautyKristi", "Thataylaa", "NikkieTutorials", "Chloe Morello", "Zoella", "Tanya Burr", "Ingrid Nilsen", "Bethany Mota", "MyLifeAsEva", "Rachel Levin", "Alisha Marie", "Aspyn Ovard", "Niki and Gabi", "Sierra Furtado", "Meredith Foster", "Superwoman", "Tyler Oakley", "Grace Helbig", "Mamrie Hart", "Hannah Hart", "Kingsley", "Connor Franta", "Ricky Dillon", "Kian Lawley", "JC Caylen", "Trevor Moran", "Andrea Russett", "Jenn McAllister", "Meghan Rienks", "Megan DeAngelis", "Alex Wassabi", "Aaron Burriss", "Gabriel Conte", "Jess Conte", "The Dolan Twins", "Ethan Dolan", "Grayson Dolan", "Colleen Ballinger", "Miranda Sings", "Todrick Hall", "PewDiePie", "Markiplier", "Jacksepticeye", "Ninja", "MrBeast", "Logan Paul", "Jake Paul", "Liza Koshy", "Lele Pons", "Jojo Siwa"]

download_4k_videos(celebrities)


