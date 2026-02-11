"""
PSD Layer Parsing Service

Parses Photoshop PSD files into semantic layout structures for content-aware
banner resizing. Implements the specification from docs/PSD_INTEGRATION.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Layer, Group

from app.models.jobs import Region


logger = logging.getLogger(__name__)


class LayerConstraints:
    """Parsed constraints from layer name modifiers."""
    
    def __init__(self, name: str):
        self.lock = "[lock]" in name.lower()
        self.flex = "[flex]" in name.lower()
        self.anchor = self._parse_anchor(name)
        self.max_scale = self._parse_scale(name, "max-scale")
        self.min_scale = self._parse_scale(name, "min-scale")
    
    def _parse_anchor(self, name: str) -> str | None:
        """Extract anchor position from [anchor:position] modifier."""
        import re
        match = re.search(r'\[anchor:([\w-]+)\]', name.lower())
        return match.group(1) if match else None
    
    def _parse_scale(self, name: str, prefix: str) -> float | None:
        """Extract scale limit from [max-scale=X] or [min-scale=X] modifier."""
        import re
        pattern = rf'\[{prefix}=([\d.]+)\]'
        match = re.search(pattern, name.lower())
        return float(match.group(1)) if match else None


class SemanticLayer:
    """Represents a parsed PSD layer with semantic classification."""
    
    def __init__(
        self,
        name: str,
        role: str,
        bbox: tuple[int, int, int, int],
        constraints: LayerConstraints,
        z_index: int,
    ):
        self.name = name
        self.role = role  # background, text, logo, product, decor, protected
        self.bbox = bbox  # (x, y, width, height)
        self.constraints = constraints
        self.z_index = z_index
    
    def to_region(self) -> Region:
        """Convert to Region model for analysis pipeline."""
        x, y, w, h = self.bbox
        return Region(
            x=x,
            y=y,
            width=w,
            height=h,
            score=1.0,
            label=self.role,
        )


class PSDParser:
    """
    Parses PSD files into semantic layout structures.
    
    Implements layer classification rules from PSD_INTEGRATION.md:
    - Background layers (bg:, background:, BG, BACKGROUND)
    - Text layers (text:, copy:, or native text layers)
    - Logo layers (logo:)
    - Product/Hero layers (product:, hero:)
    - Decorative layers (decor: or no prefix)
    - Protected layers ([protect], [lock])
    """
    
    def __init__(self, psd_path: str):
        self.psd_path = Path(psd_path)
        self.psd: PSDImage | None = None
        self.semantic_layers: List[SemanticLayer] = []
        self.background_layer: SemanticLayer | None = None
        self.warnings: List[str] = []
    
    def parse(self) -> bool:
        """
        Parse the PSD file and extract semantic layers.
        
        Returns True if parsing succeeded, False otherwise.
        """
        try:
            self.psd = PSDImage.open(self.psd_path)
            logger.info(f"Opened PSD: {self.psd_path} ({self.psd.width}x{self.psd.height})")
        except Exception as exc:
            logger.error(f"Failed to open PSD file {self.psd_path}: {exc}")
            return False
        
        # Parse all layers recursively
        self._parse_layers(self.psd, z_index=0)
        
        # Validate and identify background
        if not self._identify_background():
            logger.error("No background layer detected in PSD")
            return False
        
        # Run validation checks
        self._validate_structure()
        
        return True
    
    def _parse_layers(self, parent, z_index: int = 0) -> int:
        """
        Recursively parse layers and groups.
        
        Returns the next available z_index.
        """
        for layer in parent:
            if not layer.visible:
                continue
            
            if isinstance(layer, Group):
                # Groups inherit semantics from their name
                z_index = self._parse_group(layer, z_index)
            else:
                z_index = self._parse_layer(layer, z_index)
        
        return z_index
    
    def _parse_group(self, group: Group, z_index: int) -> int:
        """Parse a layer group and its children."""
        group_name = group.name.lower()
        
        # Check if this is a background group
        if group_name in ("bg", "background") or group_name.startswith(("bg:", "background:")):
            # Treat entire group as background
            bbox = self._get_layer_bbox(group)
            if bbox:
                constraints = LayerConstraints(group.name)
                semantic_layer = SemanticLayer(
                    name=group.name,
                    role="background",
                    bbox=bbox,
                    constraints=constraints,
                    z_index=z_index,
                )
                self.semantic_layers.append(semantic_layer)
                return z_index + 1
        
        # Otherwise, parse children with inherited constraints
        return self._parse_layers(group, z_index)
    
    def _parse_layer(self, layer: Layer, z_index: int) -> int:
        """Parse a single layer."""
        layer_name = layer.name.lower()
        constraints = LayerConstraints(layer.name)
        
        # Determine semantic role
        role = self._classify_layer(layer, layer_name, constraints)
        
        # Get bounding box
        bbox = self._get_layer_bbox(layer)
        if not bbox:
            logger.warning(f"Layer '{layer.name}' has no valid bounding box, skipping")
            return z_index
        
        semantic_layer = SemanticLayer(
            name=layer.name,
            role=role,
            bbox=bbox,
            constraints=constraints,
            z_index=z_index,
        )
        
        self.semantic_layers.append(semantic_layer)
        return z_index + 1
    
    def _classify_layer(self, layer: Layer, layer_name: str, constraints: LayerConstraints) -> str:
        """
        Classify layer into semantic role based on naming conventions.
        
        Priority order (highest to lowest):
        1. Explicitly protected ([protect], [lock])
        2. Background (bg:, background:)
        3. Text (text:, copy:, or native text layer)
        4. Logo (logo:)
        5. Product/Hero (product:, hero:)
        6. Decorative (decor: or default)
        """
        # 1. Explicitly protected
        if constraints.lock or "[protect]" in layer_name:
            return "protected"
        
        # 2. Background
        if layer_name.startswith(("bg:", "background:")):
            return "background"
        
        # 3. Text
        if layer_name.startswith(("text:", "copy:")):
            return "text"
        if layer.kind == "type":  # Native Photoshop text layer
            return "text"
        
        # 4. Logo
        if layer_name.startswith("logo:") or "logo" in layer_name:
            return "logo"
        
        # 5. Product/Hero
        if layer_name.startswith(("product:", "hero:")):
            return "product"
        
        # 6. Decorative (default)
        if layer_name.startswith("decor:"):
            return "decor"
        
        return "decor"  # Default fallback
    
    def _get_layer_bbox(self, layer) -> tuple[int, int, int, int] | None:
        """
        Extract bounding box from layer.
        
        Returns (x, y, width, height) or None if invalid.
        """
        try:
            bbox = layer.bbox
            if not bbox:
                return None
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                return None
            
            return (x1, y1, width, height)
        except Exception as exc:
            logger.warning(f"Failed to get bbox for layer '{layer.name}': {exc}")
            return None
    
    def _identify_background(self) -> bool:
        """
        Identify the background layer.
        
        Rules:
        - Exactly one background layer/group is recommended
        - If multiple exist, pick lowest z-index
        - Background must exist (FAIL if not found)
        """
        background_layers = [
            layer for layer in self.semantic_layers
            if layer.role == "background"
        ]
        
        if not background_layers:
            return False
        
        if len(background_layers) > 1:
            self.warnings.append(
                f"Multiple background layers detected ({len(background_layers)}), "
                f"using lowest z-index"
            )
            background_layers.sort(key=lambda l: l.z_index)
        
        self.background_layer = background_layers[0]
        return True
    
    def _validate_structure(self):
        """Run validation checks and generate warnings."""
        # Check for text inside background group
        if self.background_layer:
            bg_bbox = self.background_layer.bbox
            for layer in self.semantic_layers:
                if layer.role == "text" and self._bbox_contains(bg_bbox, layer.bbox):
                    self.warnings.append(
                        f"Text layer '{layer.name}' detected inside background group"
                    )
        
        # Check for protected layers overlapping canvas edge
        if self.psd:
            canvas_w, canvas_h = self.psd.width, self.psd.height
            for layer in self.semantic_layers:
                if layer.role in ("protected", "text", "logo"):
                    x, y, w, h = layer.bbox
                    if x <= 0 or y <= 0 or (x + w) >= canvas_w or (y + h) >= canvas_h:
                        self.warnings.append(
                            f"Protected layer '{layer.name}' overlaps canvas edge"
                        )
    
    def _bbox_contains(self, outer: tuple, inner: tuple) -> bool:
        """Check if outer bbox contains inner bbox."""
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return (ox <= ix and oy <= iy and 
                (ox + ow) >= (ix + iw) and (oy + oh) >= (iy + ih))
    
    def get_regions_by_role(self, role: str) -> List[Region]:
        """Get all regions matching a specific role."""
        return [
            layer.to_region()
            for layer in self.semantic_layers
            if layer.role == role
        ]
    
    def get_protected_regions(self) -> List[Region]:
        """Get all protected regions (text, logo, product, protected)."""
        protected_roles = {"text", "logo", "product", "protected"}
        return [
            layer.to_region()
            for layer in self.semantic_layers
            if layer.role in protected_roles
        ]
    
    def export_flattened_image(self, output_path: str) -> bool:
        """
        Export flattened PSD as PNG for analysis pipeline.
        
        This creates a rasterized version that can be processed by
        the existing CV-based analysis.
        """
        try:
            if not self.psd:
                return False
            
            # Composite the PSD to a PIL Image
            image = self.psd.composite()
            image.save(output_path, "PNG")
            logger.info(f"Exported flattened PSD to {output_path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to export flattened PSD: {exc}")
            return False
    
    def get_dimensions(self) -> tuple[int, int]:
        """Get PSD dimensions (width, height)."""
        if self.psd:
            return (self.psd.width, self.psd.height)
        return (0, 0)
