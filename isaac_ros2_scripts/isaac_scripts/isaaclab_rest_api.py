# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
REST API Server for Isaac Lab.

This module provides a FastAPI-based REST API server for Isaac Lab,
allowing external applications to:
- Convert URDF to USD
- Generate ArticulationCfg for Isaac Lab
- Configure and control reinforcement learning training

Usage:
    python isaaclab_rest_api.py [--port 8081] [--host 0.0.0.0]

API Endpoints:
    GET  /health                    - Health check
    POST /convert_urdf              - Convert URDF to USD
    POST /generate_config           - Generate ArticulationCfg
    POST /prepare_robot             - Convert URDF + Generate config (combined)
    POST /training/config           - Set training configuration
    GET  /training/config           - Get current training configuration
    POST /training/start            - Start training
    POST /training/stop             - Stop training
    GET  /training/status           - Get training status

API Documentation:
    http://localhost:8081/docs (Swagger UI)
"""

import os
import sys
import json
import threading
import subprocess
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn


# ============================================================================
# Data Models
# ============================================================================

class TrainingStatus(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingState:
    """Current state of Isaac Lab training."""
    status: TrainingStatus = TrainingStatus.IDLE
    config: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    current_episode: int = 0
    total_episodes: int = 0
    current_reward: float = 0.0
    best_reward: float = 0.0
    error_message: Optional[str] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False)


# Request/Response Models
class ConvertUrdfRequest(BaseModel):
    """Request model for URDF to USD conversion."""
    urdf_path: str = Field(..., description="Path to the URDF file")
    output_usd_path: Optional[str] = Field(None, description="Output USD file path")
    fixed_base: bool = Field(default=False, description="Fix robot base to world")


class GenerateConfigRequest(BaseModel):
    """Request model for ArticulationCfg generation."""
    urdf_path: str = Field(..., description="Path to the URDF file")
    usd_path: str = Field(..., description="Path to the USD file")
    output_path: Optional[str] = Field(None, description="Output Python file path")
    class_name: Optional[str] = Field(None, description="Configuration class name")


class PrepareRobotRequest(BaseModel):
    """Request model for combined URDF conversion and config generation."""
    urdf_path: str = Field(..., description="Path to the URDF file")
    output_dir: Optional[str] = Field(None, description="Output directory for generated files")
    fixed_base: bool = Field(default=False, description="Fix robot base to world")


class TrainingConfigRequest(BaseModel):
    """Request model for training configuration."""
    robot_usd_path: str = Field(..., description="Path to the robot USD file")
    robot_config_path: str = Field(..., description="Path to the ArticulationCfg Python file")
    num_envs: int = Field(default=1024, description="Number of parallel environments")
    max_episodes: int = Field(default=10000, description="Maximum training episodes")
    task_name: Optional[str] = Field(None, description="Task name (e.g., 'locomotion', 'manipulation')")
    checkpoint_dir: Optional[str] = Field(None, description="Directory to save checkpoints")
    resume_from: Optional[str] = Field(None, description="Checkpoint to resume from")
    extra_args: Dict[str, Any] = Field(default_factory=dict, description="Additional training arguments")


class ResponseModel(BaseModel):
    """Standard response model."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    current_episode: int = 0
    total_episodes: int = 0
    current_reward: float = 0.0
    best_reward: float = 0.0
    error_message: Optional[str] = None
    config: Dict[str, Any] = {}


# ============================================================================
# REST API Server
# ============================================================================

class IsaacLabRestApi:
    """REST API server for Isaac Lab control."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8081):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Isaac Lab REST API",
            description="REST API for controlling Isaac Lab training and robot preparation",
            version="1.0.0"
        )
        self.training_state = TrainingState()
        self._server_thread: Optional[threading.Thread] = None
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/health", response_model=ResponseModel)
        async def health_check():
            """Health check endpoint."""
            return ResponseModel(
                success=True,
                message="Isaac Lab REST API is running",
                data={"training_status": self.training_state.status.value}
            )

        @self.app.post("/convert_urdf", response_model=ResponseModel)
        async def convert_urdf(request: ConvertUrdfRequest):
            """
            Convert URDF to USD format.

            This endpoint uses Isaac Sim's URDF importer to convert a URDF file
            to USD format, applying custom Isaac Sim attributes.
            """
            try:
                result = self._convert_urdf(request)
                return ResponseModel(
                    success=True,
                    message=f"URDF converted to USD: {result['usd_path']}",
                    data=result
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/generate_config", response_model=ResponseModel)
        async def generate_config(request: GenerateConfigRequest):
            """
            Generate Isaac Lab ArticulationCfg from URDF.

            This endpoint parses the ros2_control tags in URDF and generates
            a Python configuration file for Isaac Lab.
            """
            try:
                result = self._generate_config(request)
                return ResponseModel(
                    success=True,
                    message=f"ArticulationCfg generated: {result['config_path']}",
                    data=result
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/prepare_robot", response_model=ResponseModel)
        async def prepare_robot(request: PrepareRobotRequest):
            """
            Prepare robot for Isaac Lab (combined URDF conversion + config generation).

            This is a convenience endpoint that:
            1. Converts URDF to USD
            2. Generates ArticulationCfg
            """
            try:
                result = self._prepare_robot(request)
                return ResponseModel(
                    success=True,
                    message="Robot prepared for Isaac Lab",
                    data=result
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/training/config", response_model=ResponseModel)
        async def set_training_config(request: TrainingConfigRequest):
            """Set training configuration."""
            if self.training_state.status == TrainingStatus.RUNNING:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot change config while training is running"
                )

            self.training_state.config = request.model_dump()
            self.training_state.total_episodes = request.max_episodes

            return ResponseModel(
                success=True,
                message="Training configuration updated",
                data=self.training_state.config
            )

        @self.app.get("/training/config", response_model=ResponseModel)
        async def get_training_config():
            """Get current training configuration."""
            return ResponseModel(
                success=True,
                message="Current training configuration",
                data=self.training_state.config
            )

        @self.app.post("/training/start", response_model=ResponseModel)
        async def start_training(background_tasks: BackgroundTasks):
            """Start Isaac Lab training."""
            if self.training_state.status == TrainingStatus.RUNNING:
                raise HTTPException(
                    status_code=409,
                    detail="Training is already running"
                )

            if not self.training_state.config:
                raise HTTPException(
                    status_code=400,
                    detail="Training configuration not set. Call /training/config first."
                )

            self.training_state.status = TrainingStatus.PREPARING
            self.training_state.start_time = datetime.now().isoformat()
            self.training_state.error_message = None

            # Start training in background
            background_tasks.add_task(self._run_training)

            return ResponseModel(
                success=True,
                message="Training started",
                data={"status": self.training_state.status.value}
            )

        @self.app.post("/training/stop", response_model=ResponseModel)
        async def stop_training():
            """Stop Isaac Lab training."""
            if self.training_state.status != TrainingStatus.RUNNING:
                raise HTTPException(
                    status_code=409,
                    detail=f"Training is not running (status: {self.training_state.status.value})"
                )

            self.training_state.status = TrainingStatus.STOPPING

            # Terminate the training process
            if self.training_state.process is not None:
                self.training_state.process.terminate()

            return ResponseModel(
                success=True,
                message="Training stop requested",
                data={"status": self.training_state.status.value}
            )

        @self.app.get("/training/status", response_model=TrainingStatusResponse)
        async def get_training_status():
            """Get current training status."""
            return TrainingStatusResponse(
                status=self.training_state.status.value,
                start_time=self.training_state.start_time,
                end_time=self.training_state.end_time,
                current_episode=self.training_state.current_episode,
                total_episodes=self.training_state.total_episodes,
                current_reward=self.training_state.current_reward,
                best_reward=self.training_state.best_reward,
                error_message=self.training_state.error_message,
                config=self.training_state.config
            )

    def _convert_urdf(self, request: ConvertUrdfRequest) -> Dict[str, Any]:
        """Convert URDF to USD using Isaac Lab standalone script or Isaac Sim."""
        # Generate output path if not provided
        if request.output_usd_path is None:
            urdf_dir = os.path.dirname(request.urdf_path)
            urdf_basename = os.path.splitext(os.path.basename(request.urdf_path))[0]
            output_usd_path = os.path.join(urdf_dir, urdf_basename + ".usd")
        else:
            output_usd_path = request.output_usd_path

        converted = False
        conversion_method = None

        # Method 1: Try Isaac Lab standalone script
        isaaclab_sh = '/workspace/isaaclab/isaaclab.sh'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, 'convert_urdf_standalone.py')

        if os.path.exists(isaaclab_sh) and os.path.exists(script_path):
            print(f"[IsaacLab API] Using standalone script for USD conversion...")
            cmd = [
                isaaclab_sh, '-p', script_path,
                '--urdf_path', request.urdf_path,
                '--output_usd_path', output_usd_path,
                '--headless',
            ]
            if request.fixed_base:
                cmd.append('--fixed_base')

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                print(f"[IsaacLab API] Standalone script output:\n{result.stdout}")
                if result.stderr:
                    print(f"[IsaacLab API] Standalone script stderr:\n{result.stderr}")

                if result.returncode == 0 and os.path.exists(output_usd_path):
                    converted = True
                    conversion_method = "Isaac Lab standalone script"
                    print(f"[IsaacLab API] Converted URDF to USD: {output_usd_path}")
                else:
                    print(f"[IsaacLab API] Standalone script failed with code {result.returncode}")
            except subprocess.TimeoutExpired:
                print("[IsaacLab API] Standalone script timed out")
            except Exception as e:
                print(f"[IsaacLab API] Standalone script failed: {e}")

        # Method 2: Try Isaac Sim's convert_urdf module
        if not converted:
            try:
                import convert_urdf
                convert_urdf.convert_urdf_to_usd(
                    urdf_path=request.urdf_path,
                    output_usd_path=output_usd_path,
                    fixed_base=request.fixed_base,
                )
                converted = True
                conversion_method = "Isaac Sim convert_urdf"
                print(f"[IsaacLab API] Converted URDF to USD using Isaac Sim: {output_usd_path}")
            except ImportError:
                print("[IsaacLab API] Isaac Sim convert_urdf not available")
            except Exception as e:
                print(f"[IsaacLab API] Isaac Sim convert_urdf failed: {e}")

        if not converted:
            raise RuntimeError(
                "USD conversion failed. Make sure to run this API in Isaac Lab or Isaac Sim environment. "
                f"isaaclab.sh exists: {os.path.exists(isaaclab_sh)}, "
                f"script exists: {os.path.exists(script_path)}"
            )

        return {
            "urdf_path": request.urdf_path,
            "usd_path": output_usd_path,
            "fixed_base": request.fixed_base,
            "conversion_method": conversion_method,
        }

    def _generate_config(self, request: GenerateConfigRequest) -> Dict[str, Any]:
        """Generate ArticulationCfg from URDF."""
        import generate_isaaclab_config

        # Generate output path if not provided
        if request.output_path is None:
            urdf_basename = os.path.splitext(os.path.basename(request.urdf_path))[0]
            output_path = f"{urdf_basename}_cfg.py"
        else:
            output_path = request.output_path

        # Generate the config
        code = generate_isaaclab_config.main(
            urdf_path=request.urdf_path,
            usd_path=request.usd_path,
            output_path=output_path,
            class_name=request.class_name,
        )

        return {
            "config_path": output_path,
            "urdf_path": request.urdf_path,
            "usd_path": request.usd_path,
        }

    def _prepare_robot(self, request: PrepareRobotRequest) -> Dict[str, Any]:
        """Combined URDF conversion and config generation."""
        # Determine output paths
        urdf_basename = os.path.splitext(os.path.basename(request.urdf_path))[0]

        if request.output_dir:
            os.makedirs(request.output_dir, exist_ok=True)
            usd_path = os.path.join(request.output_dir, urdf_basename + ".usd")
            config_path = os.path.join(request.output_dir, urdf_basename + "_cfg.py")
        else:
            urdf_dir = os.path.dirname(request.urdf_path)
            usd_path = os.path.join(urdf_dir, urdf_basename + ".usd")
            config_path = os.path.join(urdf_dir, urdf_basename + "_cfg.py")

        # Step 1: Convert URDF to USD
        convert_result = self._convert_urdf(ConvertUrdfRequest(
            urdf_path=request.urdf_path,
            output_usd_path=usd_path,
            fixed_base=request.fixed_base,
        ))
        # Update usd_path from conversion result (may have been modified)
        usd_path = convert_result.get("usd_path", usd_path)

        # Step 2: Generate config
        import generate_isaaclab_config
        code = generate_isaaclab_config.main(
            urdf_path=request.urdf_path,
            usd_path=usd_path,
            output_path=config_path,
        )

        return {
            "urdf_path": request.urdf_path,
            "usd_path": usd_path,
            "config_path": config_path,
            "conversion_method": convert_result.get("conversion_method"),
        }

    def _run_training(self):
        """Run Isaac Lab training in background."""
        try:
            self.training_state.status = TrainingStatus.RUNNING

            config = self.training_state.config

            # Build the training command
            # This assumes isaaclab CLI is available
            cmd = [
                "python", "-m", "isaaclab.train",
                "--task", config.get("task_name", "Isaac-Velocity-Flat-Anymal-C-v0"),
                "--num_envs", str(config.get("num_envs", 1024)),
                "--max_iterations", str(config.get("max_episodes", 10000)),
            ]

            if config.get("checkpoint_dir"):
                cmd.extend(["--checkpoint_dir", config["checkpoint_dir"]])

            if config.get("resume_from"):
                cmd.extend(["--resume", config["resume_from"]])

            # Add extra args
            for key, value in config.get("extra_args", {}).items():
                cmd.extend([f"--{key}", str(value)])

            print(f"[IsaacLab API] Starting training: {' '.join(cmd)}")

            # Start the training process
            self.training_state.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Monitor the process
            for line in self.training_state.process.stdout:
                print(f"[IsaacLab] {line.strip()}")
                # Parse training progress from output (simplified)
                if "Episode" in line:
                    # Try to extract episode number
                    pass

            # Wait for completion
            self.training_state.process.wait()

            if self.training_state.status == TrainingStatus.STOPPING:
                self.training_state.status = TrainingStatus.IDLE
            elif self.training_state.process.returncode == 0:
                self.training_state.status = TrainingStatus.COMPLETED
            else:
                self.training_state.status = TrainingStatus.FAILED
                self.training_state.error_message = f"Training exited with code {self.training_state.process.returncode}"

        except Exception as e:
            self.training_state.status = TrainingStatus.FAILED
            self.training_state.error_message = str(e)
            print(f"[IsaacLab API] Training failed: {e}")

        finally:
            self.training_state.end_time = datetime.now().isoformat()
            self.training_state.process = None

    def start(self):
        """Start the REST API server in a background thread."""
        def run_server():
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        print(f"[IsaacLab API] Server started at http://{self.host}:{self.port}")
        print(f"[IsaacLab API] API documentation: http://{self.host}:{self.port}/docs")

    def run(self):
        """Run the REST API server (blocking)."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Isaac Lab REST API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8081, help="Port number")

    args = parser.parse_args()

    server = IsaacLabRestApi(host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
