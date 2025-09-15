import numpy as np
import openai
import json
import re
import os
from typing import Dict, List, Callable, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key
openai.api_key = os.getenv("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")

DEBUG = False

# Existing controller functions (unchanged)
def get_move_action(observation, target_position, atol=1e-3, gain=10., close_gripper=False):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    current_position = observation['observation'][:3]

    action = gain * np.subtract(target_position, current_position)
    if close_gripper:
        gripper_action = -1.
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action

def block_is_grasped(obs, relative_grasp_position, atol=1e-3):
    return block_inside_grippers(obs, relative_grasp_position, atol=atol) and grippers_are_closed(obs, atol=atol)

def block_inside_grippers(obs, relative_grasp_position, atol=1e-3):
    gripper_position = obs['observation'][:3]
    block_position = obs['observation'][3:6]

    relative_position = np.subtract(gripper_position, block_position)

    return np.sum(np.subtract(relative_position, relative_grasp_position)**2) < atol

def grippers_are_closed(obs, atol=1e-3):
    gripper_state = obs['observation'][9:11]
    return abs(gripper_state[0] - 0.024) < atol

def grippers_are_open(obs, atol=1e-3):
    gripper_state = obs['observation'][9:11]
    return abs(gripper_state[0] - 0.05) < atol

def get_pick_and_place_control(obs, relative_grasp_position=(0., 0., -0.02), workspace_height=0.1, atol=1e-3, gain=50.):
    """
    Returns
    -------
    action : [float] * 4
    """
    gripper_position = obs['observation'][:3]
    block_position = obs['observation'][3:6]
    place_position = obs['desired_goal']

    # If the gripper is already grasping the block
    if block_is_grasped(obs, relative_grasp_position, atol=atol):

        # If the block is already at the place position, do nothing except keep the gripper closed
        if np.sum(np.subtract(block_position, place_position)**2) < 1e-6:
            if DEBUG:
                print("The block is already at the place position; do nothing")
            return np.array([0., 0., 0., -1.])

        # Move to the place position while keeping the gripper closed
        target_position = np.add(place_position, relative_grasp_position)
        target_position[2] += workspace_height/2.
        if DEBUG:
            print("Move to above the place position")
        return get_move_action(obs, target_position, atol=atol, gain=gain, close_gripper=True)

    # If the block is ready to be grasped
    if block_inside_grippers(obs, relative_grasp_position, atol=atol):

        # Close the grippers
        if DEBUG:
            print("Close the grippers")
        return np.array([0., 0., 0., -1.])

    # If the gripper is above the block
    if (gripper_position[0] - block_position[0])**2 + (gripper_position[1] - block_position[1])**2 < atol * 5e-5:

        # If the grippers are closed, open them
        if not grippers_are_open(obs, atol=atol):
            if DEBUG:
                print("Open the grippers")
            return np.array([0., 0., 0., 1.])

        # Move down to grasp
        target_position = np.add(block_position, relative_grasp_position)
        if DEBUG:
            print("Move down to grasp")
        return get_move_action(obs, target_position, atol=atol, gain=10)

    # Else move the gripper to above the block
    target_position = np.add(block_position, relative_grasp_position)
    target_position[2] += workspace_height
    if DEBUG:
        print("Move to above the block")
    return get_move_action(obs, target_position, atol=atol, gain=10)

# =============================================================================
# Language Model Integration Layer (Self-Refine Approach)
# =============================================================================

class LanguageControllerInterface:
    """Interface for converting natural language to robot control actions"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.model = model
        self.few_shot_examples = self._load_few_shot_examples()
        self.previous_attempts = []
        
        # Set API key (use provided key or environment variable)
        if api_key:
            openai.api_key = api_key
        elif openai.api_key is None:
            logger.warning("No OpenAI API key provided. Language features will be limited.")
    
    def _load_few_shot_examples(self) -> List[Dict]:
        """Load few-shot examples for language-to-control mapping"""
        return [
            {
                "language": "pick up the block and place it at the target",
                "description": "Standard pick and place operation with grasp and release",
                "controller_function": "get_pick_and_place_control",
                "parameters": {
                    "relative_grasp_position": [0.0, 0.0, -0.02],
                    "workspace_height": 0.1,
                    "gain": 50.0,
                    "atol": 1e-3
                }
            },
            {
                "language": "gently pick up the block",
                "description": "Gentle grasping with lower gain for precise movement",
                "controller_function": "get_pick_and_place_control",
                "parameters": {
                    "relative_grasp_position": [0.0, 0.0, -0.015],
                    "workspace_height": 0.08,
                    "gain": 25.0,
                    "atol": 5e-4
                }
            },
            {
                "language": "quickly move the block to target",
                "description": "Fast movement with higher gain",
                "controller_function": "get_pick_and_place_control",
                "parameters": {
                    "relative_grasp_position": [0.0, 0.0, -0.02],
                    "workspace_height": 0.12,
                    "gain": 75.0,
                    "atol": 2e-3
                }
            },
            {
                "language": "precisely position the block",
                "description": "High precision placement with tight tolerance",
                "controller_function": "get_pick_and_place_control",
                "parameters": {
                    "relative_grasp_position": [0.0, 0.0, -0.01],
                    "workspace_height": 0.07,
                    "gain": 20.0,
                    "atol": 1e-4
                }
            }
        ]
    
    def language_to_control(self, language_command: str, max_iterations: int = 3) -> Dict:
        """
        Convert natural language command to control parameters using Self-Refine approach
        """
        logger.info(f"Processing language command: {language_command}")
        
        # Check if API key is available
        if openai.api_key is None:
            logger.warning("No API key available. Using default configuration.")
            return self._get_default_config()
        
        best_result = None
        best_score = -1
        self.previous_attempts = []
        
        for iteration in range(max_iterations):
            logger.info(f"Self-Refine iteration {iteration + 1}/{max_iterations}")
            
            # Generate controller parameters
            controller_config = self._generate_controller_config(language_command, iteration)
            
            if not controller_config:
                continue
            
            # Store attempt for refinement
            self.previous_attempts.append(controller_config)
            
            # Test the configuration (simulated or real)
            score = self._evaluate_controller_config(controller_config, language_command)
            
            if score > best_score:
                best_score = score
                best_result = controller_config
            
            logger.info(f"Iteration {iteration + 1} score: {score:.2f}")
            
            # If we have a good enough result, break early
            if best_score >= 0.8:
                break
        
        if best_result:
            logger.info(f"Best controller configuration found with score: {best_score:.2f}")
            return best_result
        else:
            logger.warning("No suitable controller configuration found")
            return self._get_default_config()
    
    def _generate_controller_config(self, language_command: str, iteration: int) -> Dict:
        """Generate controller configuration using LLM"""
        
        prompt = self._build_prompt(language_command, iteration)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a robotics control expert. Map natural language commands to controller parameters for a Fetch robot. Respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7 * (iteration + 1) / 3,  # Increase temperature with iterations
                max_tokens=500
            )
            
            generated_text = response.choices[0].message.content
            return self._parse_generated_config(generated_text)
            
        except Exception as e:
            logger.error(f"Error generating controller config: {e}")
            return None
    
    def _build_prompt(self, language_command: str, iteration: int) -> str:
        """Build prompt for LLM based on iteration"""
        
        examples_text = "\n".join([
            f"Command: {ex['language']}\n"
            f"Function: {ex['controller_function']}\n"
            f"Params: {json.dumps(ex['parameters'], indent=2)}\n"
            for ex in self.few_shot_examples[:2]  # Use first 2 examples for brevity
        ])
        
        if iteration == 0:
            # Initial generation
            return f"""
Based on these examples, map the natural language command to controller parameters for a Fetch robot:

Examples:
{examples_text}

Available controller functions: get_pick_and_place_control
Available parameters: relative_grasp_position, workspace_height, gain, atol

Command: {language_command}

Respond with JSON only in this exact format:
{{
    "controller_function": "function_name",
    "parameters": {{
        "param1": value1,
        "param2": value2
    }},
    "rationale": "brief explanation"
}}
"""
        else:
            # Refinement iteration
            previous_config = self.previous_attempts[-1]
            return f"""
Refine the controller parameters for better performance. The previous attempt was:

Previous configuration:
{json.dumps(previous_config, indent=2)}

Command: {language_command}

Suggest improved parameters considering the task requirements. Focus on adjusting:
- relative_grasp_position (3D offset for grasping)
- workspace_height (clearance height)
- gain (movement speed/aggression)
- atol (tolerance/precision)

Respond with JSON only in the same format.
"""
    
    def _parse_generated_config(self, generated_text: str) -> Dict:
        """Parse LLM response into controller configuration"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                config = json.loads(json_match.group())
                
                # Validate required fields
                if ('controller_function' in config and 
                    'parameters' in config and
                    config['controller_function'] in globals()):
                    
                    # Ensure parameters have correct types
                    if 'relative_grasp_position' in config['parameters']:
                        config['parameters']['relative_grasp_position'] = tuple(
                            config['parameters']['relative_grasp_position']
                        )
                    
                    return config
                    
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Generated text: {generated_text}")
        
        return None
    
    def _evaluate_controller_config(self, config: Dict, language_command: str) -> float:
        """
        Evaluate controller configuration (simulated evaluation)
        In real implementation, this would run actual simulations
        """
        # Simulated evaluation based on language command features
        score = 0.5  # Base score
        
        # Score based on keyword matching with language command
        keyword_features = {
            "gently": {
                "gain": lambda x: 1.0 - min(abs(x - 25) / 50, 1.0),
                "workspace_height": lambda x: 1.0 - min(abs(x - 0.08) / 0.12, 1.0)
            },
            "quickly": {
                "gain": lambda x: 1.0 - min(abs(x - 75) / 50, 1.0),
                "workspace_height": lambda x: 1.0 - min(abs(x - 0.12) / 0.12, 1.0)
            },
            "precisely": {
                "atol": lambda x: 1.0 - min(abs(x - 1e-4) / 1e-3, 1.0),
                "gain": lambda x: 1.0 - min(abs(x - 20) / 50, 1.0)
            },
            "secure": {
                "relative_grasp_position": lambda x: 1.0 - min(abs(x[2] + 0.02) / 0.02, 1.0)
            },
            "carefully": {
                "gain": lambda x: 1.0 - min(abs(x - 30) / 50, 1.0),
                "atol": lambda x: 1.0 - min(abs(x - 5e-4) / 1e-3, 1.0)
            }
        }
        
        command_lower = language_command.lower()
        for keyword, param_weights in keyword_features.items():
            if keyword in command_lower:
                for param, weight_func in param_weights.items():
                    if param in config['parameters']:
                        param_value = config['parameters'][param]
                        try:
                            score += 0.15 * weight_func(param_value)
                        except (TypeError, IndexError):
                            pass
        
        return min(max(score, 0.0), 1.0)
    
    def _get_default_config(self) -> Dict:
        """Get default controller configuration"""
        return {
            "controller_function": "get_pick_and_place_control",
            "parameters": {
                "relative_grasp_position": (0.0, 0.0, -0.02),
                "workspace_height": 0.1,
                "gain": 50.0,
                "atol": 1e-3
            },
            "rationale": "Default pick and place configuration"
        }

# =============================================================================
# Main Interface Function
# =============================================================================

class NaturalLanguageController:
    """Main interface for natural language control of Fetch robot"""
    
    def __init__(self, api_key: str = None):
        self.language_interface = LanguageControllerInterface(api_key)
        self.current_config = None
    
    def process_command(self, language_command: str, observation: Dict) -> np.ndarray:
        """
        Process natural language command and return control action
        """
        logger.info(f"Processing command: {language_command}")
        
        # Get controller configuration from language
        self.current_config = self.language_interface.language_to_control(language_command)
        
        if not self.current_config:
            logger.warning("Using default controller configuration")
            self.current_config = self.language_interface._get_default_config()
        
        # Execute the controller function with parameters
        controller_func = globals()[self.current_config['controller_function']]
        parameters = self.current_config['parameters']
        
        try:
            action = controller_func(observation, **parameters)
            logger.info(f"Generated action: {action}")
            return action
        except Exception as e:
            logger.error(f"Error executing controller: {e}")
            # Fallback to default action
            return np.array([0., 0., 0., 0.])
    
    def get_current_config(self) -> Dict:
        """Get the current controller configuration"""
        return self.current_config

# =============================================================================
# Demonstration and Usage Examples
# =============================================================================

def demonstrate_language_control():
    """Demonstrate the language-to-control functionality"""
    
    # Initialize the natural language controller with API key
    nl_controller = NaturalLanguageController()
    
    # Example observations (simulated)
    example_observation = {
        'observation': np.array([0.5, 0.3, 0.2, 0.6, 0.4, 0.3, 0, 0, 0, 0.05, 0.05]),
        'desired_goal': np.array([0.7, 0.5, 0.1])
    }
    
    # Test different language commands
    test_commands = [
        "pick up the block and place it at the target",
        "gently grasp the block and move it",
        "quickly transport the block to the goal position",
        "precisely position the block at the target",
        "carefully pick up the blue block and secure it"
    ]
    
    print("Testing Natural Language to Control Conversion")
    print("=" * 60)
    
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        print("-" * 40)
        
        action = nl_controller.process_command(command, example_observation)
        config = nl_controller.get_current_config()
        
        print(f"Controller Function: {config['controller_function']}")
        print(f"Parameters: {json.dumps(config['parameters'], indent=2)}")
        print(f"Generated Action: {action}")
        print(f"Rationale: {config.get('rationale', 'N/A')}")

# =============================================================================
# Integration with RPL Framework
# =============================================================================

def create_nl_integrated_controller(api_key: str = None):
    """
    Create a controller that can be integrated with RPL framework
    Returns a function with the same signature as original controllers
    """
    nl_controller = NaturalLanguageController(api_key)
    
    def integrated_controller(obs, **kwargs):
        # Get language command from kwargs or use default
        language_command = kwargs.get('language_command', 'pick up the block and place it at the target')
        return nl_controller.process_command(language_command, obs)
    
    return integrated_controller

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Check if API key is set
    if openai.api_key:
        print("OpenAI API key is set successfully")
        print("Key prefix:", openai.api_key[:20] + "..." if openai.api_key else "None")
    else:
        print("Warning: No OpenAI API key found")
        print("Language features will use default configurations only")
    
    # Demonstrate the functionality
    demonstrate_language_control()
    
    # Example of how to integrate with existing RPL code
    print(f"\n{'='*60}")
    print("RPL Integration Example:")
    print(f"{'='*60}")
    
    # Create an integrated controller
    nl_controlled_function = create_nl_integrated_controller()
    
    # Test it with an observation (simulating RPL usage)
    test_obs = {
        'observation': np.array([0.5, 0.3, 0.2, 0.6, 0.4, 0.3, 0, 0, 0, 0.05, 0.05]),
        'desired_goal': np.array([0.7, 0.5, 0.1])
    }
    
    # Test with different language commands
    result1 = nl_controlled_function(test_obs, language_command="gently pick up the block")
    result2 = nl_controlled_function(test_obs, language_command="quickly move to target")
    
    print(f"Gentle action: {result1}")
    print(f"Quick action: {result2}")
    
    print("\nIntegration ready for RPL framework!")