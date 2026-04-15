import yt_dlp
import sys
import os

def download_video(url, output_path="assets/videos"):
    """
    Downloads a YouTube video or Short using yt-dlp.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # yt-dlp options
    ydl_opts = {
        # Best video and best audio, merged into mp4
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        # Set the output filename template
        'outtmpl': os.path.join(output_path, '%(title)s_%(id)s.%(ext)s'),
        # Ensure the final container is mp4
        'merge_output_format': 'mp4',
        # Ignore errors and continue
        'ignoreerrors': True,
        # Print progress
        'progress_hooks': [progress_hook]
    }

    print(f"Starting download for: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download([url])
        if error_code == 0:
            print(f"\nDownload completed successfully! Saved in '{output_path}'")
        else:
            print("\nErrors occurred during download.")

def progress_hook(d):
    """
    Hook to print download progress.
    """
    if d['status'] == 'finished':
        print(f"\nDownload finished, now converting/merging...")
    elif d['status'] == 'downloading':
        percent_str = d.get('_percent_str', 'N/A')
        speed_str = d.get('_speed_str', 'N/A')
        print(f"Downloading: {percent_str} (Speed: {speed_str})", end='\r')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run --with yt-dlp python3 download_youtube.py <YouTube_URL>")
        sys.exit(1)
        
    video_url = sys.argv[1]
    download_video(video_url)
