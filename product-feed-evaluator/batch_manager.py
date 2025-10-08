"""
Batch Management Utility for Product Feed Evaluator

This script provides utilities to manage batch files, resume evaluations,
and handle large dataset processing more effectively.
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd


class BatchManager:
    """Manages batch files and provides utilities for large dataset processing."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def list_batch_sessions(self) -> List[Dict[str, Any]]:
        """List all batch processing sessions."""
        sessions = []
        
        # Group files by timestamp
        timestamp_groups = {}
        for file_path in self.output_dir.glob("batch_*.json"):
            parts = file_path.stem.split("_")
            if len(parts) >= 3:
                timestamp = "_".join(parts[2:])  # Everything after batch_XXXX_
                if timestamp not in timestamp_groups:
                    timestamp_groups[timestamp] = []
                timestamp_groups[timestamp].append(file_path)
        
        for timestamp, files in timestamp_groups.items():
            # Sort files by batch number
            files.sort(key=lambda x: int(x.stem.split("_")[1]))
            
            # Get progress info
            progress_file = self.output_dir / f"progress_{timestamp}.json"
            progress_data = None
            if progress_file.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                except:
                    pass
            
            sessions.append({
                'timestamp': timestamp,
                'batch_files': files,
                'progress_data': progress_data,
                'total_batches': len(files)
            })
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
    
    def get_session_info(self, timestamp: str) -> Dict[str, Any]:
        """Get detailed information about a specific session."""
        session_info = {
            'timestamp': timestamp,
            'batch_files': [],
            'total_results': 0,
            'progress_data': None,
            'status': 'unknown'
        }
        
        # Get batch files
        batch_files = list(self.output_dir.glob(f"batch_*_{timestamp}.json"))
        batch_files.sort(key=lambda x: int(x.stem.split("_")[1]))
        session_info['batch_files'] = batch_files
        
        # Count total results
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    session_info['total_results'] += len(batch_data)
            except:
                pass
        
        # Get progress data
        progress_file = self.output_dir / f"progress_{timestamp}.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    session_info['progress_data'] = json.load(f)
            except:
                pass
        
        # Determine status
        if session_info['progress_data']:
            last_processed = session_info['progress_data'].get('last_processed', 0)
            total = session_info['progress_data'].get('total', 0)
            if last_processed >= total:
                session_info['status'] = 'completed'
            else:
                session_info['status'] = 'in_progress'
        else:
            session_info['status'] = 'completed' if session_info['total_results'] > 0 else 'unknown'
        
        return session_info
    
    def consolidate_session(self, timestamp: str) -> pd.DataFrame:
        """Consolidate all batch files from a session into a single DataFrame."""
        session_info = self.get_session_info(timestamp)
        all_results = []
        
        for batch_file in session_info['batch_files']:
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    all_results.extend(batch_data)
            except Exception as e:
                print(f"Error loading batch file {batch_file}: {e}")
        
        return pd.DataFrame(all_results)
    
    def cleanup_session(self, timestamp: str) -> bool:
        """Clean up all files for a specific session."""
        try:
            # Remove batch files
            batch_files = list(self.output_dir.glob(f"batch_*_{timestamp}.json"))
            for batch_file in batch_files:
                batch_file.unlink()
            
            # Remove progress file
            progress_file = self.output_dir / f"progress_{timestamp}.json"
            if progress_file.exists():
                progress_file.unlink()
            
            return True
        except Exception as e:
            print(f"Error cleaning up session {timestamp}: {e}")
            return False
    
    def cleanup_all_sessions(self) -> bool:
        """Clean up all batch files and sessions."""
        try:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
                self.output_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"Error cleaning up all sessions: {e}")
            return False
    
    def get_latest_session(self) -> Optional[Dict[str, Any]]:
        """Get the most recent session."""
        sessions = self.list_batch_sessions()
        return sessions[0] if sessions else None
    
    def resume_latest_session(self) -> Optional[pd.DataFrame]:
        """Resume the most recent session."""
        latest_session = self.get_latest_session()
        if latest_session:
            return self.consolidate_session(latest_session['timestamp'])
        return None


def main():
    """Command-line interface for batch management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage batch processing files")
    parser.add_argument("--list", action="store_true", help="List all batch sessions")
    parser.add_argument("--info", type=str, help="Get info about a specific session")
    parser.add_argument("--consolidate", type=str, help="Consolidate a session into CSV")
    parser.add_argument("--cleanup", type=str, help="Clean up a specific session")
    parser.add_argument("--cleanup-all", action="store_true", help="Clean up all sessions")
    parser.add_argument("--resume", action="store_true", help="Resume latest session")
    
    args = parser.parse_args()
    manager = BatchManager()
    
    if args.list:
        sessions = manager.list_batch_sessions()
        print(f"Found {len(sessions)} batch sessions:")
        for session in sessions:
            info = manager.get_session_info(session['timestamp'])
            print(f"  {session['timestamp']}: {info['total_results']} results, {info['status']}")
    
    elif args.info:
        info = manager.get_session_info(args.info)
        print(f"Session: {info['timestamp']}")
        print(f"Status: {info['status']}")
        print(f"Total results: {info['total_results']}")
        print(f"Batch files: {len(info['batch_files'])}")
        if info['progress_data']:
            print(f"Progress: {info['progress_data'].get('last_processed', 0)}/{info['progress_data'].get('total', 0)}")
    
    elif args.consolidate:
        df = manager.consolidate_session(args.consolidate)
        output_file = f"consolidated_{args.consolidate}.csv"
        df.to_csv(output_file, index=False)
        print(f"Consolidated {len(df)} results to {output_file}")
    
    elif args.cleanup:
        success = manager.cleanup_session(args.cleanup)
        print(f"Cleanup {'successful' if success else 'failed'}")
    
    elif args.cleanup_all:
        success = manager.cleanup_all_sessions()
        print(f"Cleanup all {'successful' if success else 'failed'}")
    
    elif args.resume:
        df = manager.resume_latest_session()
        if df is not None:
            print(f"Resumed session with {len(df)} results")
            output_file = f"resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")
        else:
            print("No session found to resume")


if __name__ == "__main__":
    main()