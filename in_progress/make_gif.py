#!/usr/bin/env python3
"""
Create animated GIF from PNG files in a folder.

Usage:
    python make_gif.py <folder_path> [options]

Examples:
    python make_gif.py vector_field_plots/lambda_1_00e01/
    python make_gif.py vector_field_plots/lambda_1_00e01/ --duration 0.2 --output my_animation.gif
    python make_gif.py vector_field_plots/lambda_1_00e01/ --pattern "vector_field_epoch_*.png" --loop 1
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import re
from PIL import Image
import numpy as np


def natural_sort_key(text):
    """
    Sort key for natural sorting (e.g., epoch_1, epoch_2, ..., epoch_10, epoch_11)
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


def create_gif_from_folder(folder_path, output_path=None, pattern="*.png", duration=0.5, 
                          loop=0, resize=None, exclude_patterns=None, include_initial=True,
                          include_final=True, verbose=True):
    """
    Create animated GIF from PNG files in a folder.
    
    Args:
        folder_path (str): Path to folder containing PNG files
        output_path (str, optional): Output GIF path. If None, auto-generates name.
        pattern (str): Glob pattern to match files (default: "*.png")
        duration (float): Duration between frames in seconds (default: 0.5)
        loop (int): Number of loops (0 = infinite, default: 0)
        resize (tuple, optional): Resize images to (width, height)
        exclude_patterns (list, optional): List of patterns to exclude
        include_initial (bool): Include files with "initial" in name (default: True)
        include_final (bool): Include files with "final" in name (default: True)
        verbose (bool): Print progress messages (default: True)
        
    Returns:
        str: Path to created GIF file
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Find PNG files
    png_files = list(folder_path.glob(pattern))
    
    if not png_files:
        raise ValueError(f"No PNG files found matching pattern '{pattern}' in {folder_path}")
    
    # Apply exclusion patterns
    if exclude_patterns:
        for exclude_pattern in exclude_patterns:
            png_files = [f for f in png_files if exclude_pattern not in f.name]
    
    # Filter initial/final files based on flags
    if not include_initial:
        png_files = [f for f in png_files if "initial" not in f.name.lower()]
    
    if not include_final:
        png_files = [f for f in png_files if "final" not in f.name.lower()]
    
    # Sort files naturally (handles epoch numbers correctly)
    png_files.sort(key=lambda x: natural_sort_key(x.name))
    
    if verbose:
        print(f"📁 Found {len(png_files)} PNG files in {folder_path}")
        print(f"🔄 Creating GIF with {duration}s per frame, loop={loop}")
        if resize:
            print(f"📐 Resizing images to {resize}")
    
    # Load and process images
    images = []
    for i, png_file in enumerate(png_files):
        try:
            img = Image.open(png_file)
            
            # Convert to RGB if necessary (GIFs don't support RGBA)
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if specified
            if resize:
                img = img.resize(resize, Image.Resampling.LANCZOS)
            
            images.append(img)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(png_files)} images...")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not process {png_file}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images could be processed")
    
    # Generate output path if not specified
    if output_path is None:
        folder_name = folder_path.name
        output_path = folder_path / f"{folder_name}_animation.gif"
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create GIF
    if verbose:
        print(f"🎬 Saving GIF with {len(images)} frames...")
    
    # Convert duration from seconds to milliseconds
    duration_ms = int(duration * 1000)
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True
    )
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    if verbose:
        print(f"✅ GIF created successfully!")
        print(f"   📄 Output: {output_path}")
        print(f"   📊 Frames: {len(images)}")
        print(f"   ⏱️  Duration: {duration}s per frame")
        print(f"   💾 Size: {file_size_mb:.2f} MB")
        
        # Show first few and last few filenames for verification
        if len(png_files) > 6:
            print(f"   🗂️  Files: {png_files[0].name}, {png_files[1].name}, ..., {png_files[-2].name}, {png_files[-1].name}")
        else:
            print(f"   🗂️  Files: {', '.join([f.name for f in png_files])}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIF from PNG files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s vector_field_plots/lambda_1_00e01/
  %(prog)s vector_field_plots/lambda_1_00e01/ --duration 0.2 --output my_animation.gif
  %(prog)s . --pattern "epoch_*.png" --exclude comparison --no-initial --no-final
  %(prog)s plots/ --resize 800 600 --duration 0.1 --loop 3
        """
    )
    
    parser.add_argument(
        "folder_path", 
        help="Path to folder containing PNG files"
    )
    
    parser.add_argument(
        "-o", "--output", 
        help="Output GIF file path (default: auto-generate based on folder name)"
    )
    
    parser.add_argument(
        "-p", "--pattern", 
        default="*.png",
        help="Glob pattern to match files (default: '*.png')"
    )
    
    parser.add_argument(
        "-d", "--duration", 
        type=float, 
        default=0.5,
        help="Duration between frames in seconds (default: 0.5)"
    )
    
    parser.add_argument(
        "-l", "--loop", 
        type=int, 
        default=0,
        help="Number of loops (0 = infinite, default: 0)"
    )
    
    parser.add_argument(
        "-r", "--resize", 
        nargs=2, 
        type=int, 
        metavar=("WIDTH", "HEIGHT"),
        help="Resize images to WIDTH HEIGHT (e.g., --resize 800 600)"
    )
    
    parser.add_argument(
        "-e", "--exclude", 
        action="append",
        help="Exclude files containing this pattern (can be used multiple times)"
    )
    
    parser.add_argument(
        "--no-initial", 
        action="store_true",
        help="Exclude files with 'initial' in the name"
    )
    
    parser.add_argument(
        "--no-final", 
        action="store_true",
        help="Exclude files with 'final' in the name"
    )
    
    parser.add_argument(
        "-q", "--quiet", 
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    try:
        # Convert resize to tuple if provided
        resize = tuple(args.resize) if args.resize else None
        
        gif_path = create_gif_from_folder(
            folder_path=args.folder_path,
            output_path=args.output,
            pattern=args.pattern,
            duration=args.duration,
            loop=args.loop,
            resize=resize,
            exclude_patterns=args.exclude,
            include_initial=not args.no_initial,
            include_final=not args.no_final,
            verbose=not args.quiet
        )
        
        print(f"🎉 Success! GIF saved to: {gif_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()