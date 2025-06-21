#!/usr/bin/env python3
"""
SceneValidator - A tool for validating scene descriptions against predefined rules.

This module provides functionality to validate JSON scene descriptions against
a set of customizable rules, ensuring they meet required standards before
proceeding to production.
"""

import json
import logging
import os
import datetime
from typing import Dict, List, Any, Optional, Union

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.validation_status = "PENDING"
        self.issues = []
        self.valid_elements = 0
        self.invalid_elements = 0
        self.validated_scene = {}
        self.timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    def add_issue(self, severity: str, element: str, message: str) -> None:
        """Add a validation issue."""
        self.issues.append({
            "severity": severity,
            "element": element,
            "message": message
        })
        self.invalid_elements += 1
    
    def set_status(self, status: str) -> None:
        """Set the validation status."""
        self.validation_status = status
    
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.validation_status == "PASSED"
    
    def get_issues(self) -> List[Dict[str, str]]:
        """Get all validation issues."""
        return self.issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scene_id": self.scene_id,
            "validation_status": self.validation_status,
            "issues": self.issues,
            "valid_elements": self.valid_elements,
            "invalid_elements": self.invalid_elements,
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class RuleEngine:
    """Applies validation rules to scene descriptions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_required_elements(self, scene: Dict[str, Any], result: ValidationResult) -> None:
        """Validate that all required elements are present."""
        required_elements = self.config.get("validation_rules", {}).get("required_elements", [])
        
        for element in required_elements:
            if element not in scene:
                result.add_issue(
                    "ERROR",
                    element,
                    f"Required element '{element}' is missing"
                )
            else:
                result.valid_elements += 1
    
    def validate_technical_requirements(self, scene: Dict[str, Any], result: ValidationResult) -> None:
        """Validate technical requirements like resolution and color profile."""
        tech_reqs = scene.get("technical_requirements", {})
        valid_resolutions = self.config.get("validation_rules", {}).get(
            "technical_requirements", {}).get("resolutions", [])
        valid_color_profiles = self.config.get("validation_rules", {}).get(
            "technical_requirements", {}).get("color_profiles", [])
        
        # Check resolution
        if "minimum_resolution" in tech_reqs:
            resolution = tech_reqs["minimum_resolution"]
            if resolution not in valid_resolutions:
                result.add_issue(
                    "WARNING",
                    "technical_requirements.minimum_resolution",
                    f"Resolution '{resolution}' is not in the list of standard resolutions: {', '.join(valid_resolutions)}"
                )
            else:
                result.valid_elements += 1
        
        # Check color profile
        if "color_profile" in tech_reqs:
            color_profile = tech_reqs["color_profile"]
            if color_profile not in valid_color_profiles:
                result.add_issue(
                    "ERROR",
                    "technical_requirements.color_profile",
                    f"Color profile '{color_profile}' does not match project standards: {', '.join(valid_color_profiles)}"
                )
            else:
                result.valid_elements += 1
    
    def validate_elements(self, scene: Dict[str, Any], result: ValidationResult) -> None:
        """Validate scene elements like props and characters."""
        elements = scene.get("elements", [])
        
        for i, element in enumerate(elements):
            # Validate element type
            if "type" not in element:
                result.add_issue(
                    "ERROR",
                    f"elements[{i}]",
                    "Element is missing required 'type' field"
                )
                continue
            
            # Validate required elements
            if "required" in element and element["required"] and "name" not in element:
                result.add_issue(
                    "ERROR",
                    f"elements[{i}]",
                    "Required element is missing 'name' field"
                )
            else:
                result.valid_elements += 1
            
            # Example of position validation
            if element.get("type") == "character" and element.get("position") == "left":
                # This would normally check against camera blocking rules from config
                result.add_issue(
                    "WARNING",
                    f"elements[{i}]",
                    f"Character '{element.get('name', 'unknown')}' position conflicts with camera blocking"
                )
    
    def apply_custom_rules(self, scene: Dict[str, Any], result: ValidationResult) -> None:
        """Apply custom validation rules defined in config."""
        # This would load and execute custom rule scripts
        # For demonstration, we'll just log that this would happen
        custom_rules = self.config.get("validation_rules", {}).get("custom_rules", [])
        for rule in custom_rules:
            self.logger.info(f"Would apply custom rule: {rule.get('name')}")    


class SceneValidator:
    """Main validator class for scene descriptions."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_file)
        self.rule_engine = RuleEngine(self.config)
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        # In a real implementation, this would load from a YAML file
        # For demonstration, we'll return a hardcoded config
        return {
            "validation_rules": {
                "required_elements": ["scene_id", "description"],
                "technical_requirements": {
                    "resolutions": ["1920x1080", "3840x2160"],
                    "color_profiles": ["sRGB", "Rec709"]
                },
                "custom_rules": [
                    {"name": "character_position", "script": "rules/character_position.py"}
                ]
            }
        }
    
    def validate(self, scene_file: str) -> ValidationResult:
        """Validate a scene from a file."""
        try:
            with open(scene_file, 'r') as f:
                scene_data = json.load(f)
            return self.validate_json(scene_data)
        except FileNotFoundError:
            self.logger.error(f"Scene file not found: {scene_file}")
            result = ValidationResult("unknown")
            result.set_status("FAILED")
            result.add_issue("ERROR", "file", f"Scene file not found: {scene_file}")
            return result
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in scene file: {scene_file}")
            result = ValidationResult("unknown")
            result.set_status("FAILED")
            result.add_issue("ERROR", "file", f"Invalid JSON in scene file: {scene_file}")
            return result
    
    def validate_json(self, scene_data: Dict[str, Any]) -> ValidationResult:
        """Validate a scene from JSON data."""
        scene_id = scene_data.get("scene_id", "unknown")
        result = ValidationResult(scene_id)
        
        # Trigger validation started event
        self._trigger_event("on_validation_started", scene_id)
        
        # Apply validation rules
        self.rule_engine.validate_required_elements(scene_data, result)
        self.rule_engine.validate_technical_requirements(scene_data, result)
        self.rule_engine.validate_elements(scene_data, result)
        self.rule_engine.apply_custom_rules(scene_data, result)
        
        # Set final status
        if result.invalid_elements == 0:
            result.set_status("PASSED")
        else:
            result.set_status("FAILED")
        
        # Store validated scene
        result.validated_scene = scene_data
        
        # Trigger validation complete event
        self._trigger_event("on_validation_complete", result)
        
        return result
    
    def set_log_level(self, level: str) -> None:
        """Set the log level for the validator."""
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        logging.getLogger(__name__).setLevel(numeric_level)
    
    def validate_schema_only(self, scene_data: Dict[str, Any]) -> bool:
        """Validate only the JSON schema of a scene."""
        # This would normally use a JSON Schema validator
        # For demonstration, we'll just check for required fields
        required_elements = self.config.get("validation_rules", {}).get("required_elements", [])
        return all(element in scene_data for element in required_elements)
    
    def validate_example(self) -> ValidationResult:
        """Validate a built-in example scene."""
        example_scene = {
            "scene_id": "example_001",
            "description": "Example scene for testing",
            "elements": [
                {
                    "type": "prop",
                    "name": "coffee table",
                    "position": "center",
                    "required": True
                },
                {
                    "type": "character",
                    "name": "John",
                    "position": "left",
                    "required": True
                }
            ],
            "technical_requirements": {
                "minimum_resolution": "1920x1080",
                "color_profile": "sRGB",
                "audio_channels": 2
            }
        }
        return self.validate_json(example_scene)
    
    def _trigger_event(self, event_name: str, data: Any) -> None:
        """Trigger an event with the given name and data."""
        # In a real implementation, this would use an event system
        # For demonstration, we'll just log the event
        self.logger.debug(f"Event triggered: {event_name}")


class ReportGenerator:
    """Generates detailed validation reports."""
    
    def __init__(self, result: ValidationResult):
        self.result = result
    
    def generate_text_report(self) -> str:
        """Generate a text-based report."""
        lines = [
            f"Validation Report for Scene: {self.result.scene_id}",
            f"Status: {self.result.validation_status}",
            f"Timestamp: {self.result.timestamp}",
            f"Valid Elements: {self.result.valid_elements}",
            f"Invalid Elements: {self.result.invalid_elements}",
            "\nIssues:"
        ]
        
        if not self.result.issues:
            lines.append("  No issues found.")
        else:
            for issue in self.result.issues:
                lines.append(f"  [{issue['severity']}] {issue['element']}: {issue['message']}")
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> str:
        """Generate a JSON report."""
        return self.result.to_json()
    
    def save_report(self, filename: str, format_type: str = "text") -> None:
        """Save the report to a file."""
        if format_type.lower() == "json":
            content = self.generate_json_report()
        else:
            content = self.generate_text_report()
        
        with open(filename, 'w') as f:
            f.write(content)


def main():
    """Example usage of the SceneValidator."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create validator
    validator = SceneValidator()
    
    # Validate example scene
    result = validator.validate_example()
    
    # Generate and print report
    report_gen = ReportGenerator(result)
    print(report_gen.generate_text_report())


if __name__ == "__main__":
    main()
