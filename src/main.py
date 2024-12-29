import streamlit as st
import pygame
import sys
import math
import numpy as np
from scipy.spatial import Delaunay
# ... other imports ...

class DelaunayDemo:
    def __init__(self):
        pygame.init()
        
        # Check if running in Streamlit
        self.is_streamlit = 'streamlit' in sys.modules
        
        if not self.is_streamlit:
            # Normal Pygame initialization for PyScript
            pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
            self.screen = pygame.display.get_surface()
        else:
            # Streamlit initialization
            self.screen = pygame.Surface((WIDTH, HEIGHT))
            
        # ... rest of your initialization code ...

    def draw(self):
        """Draw everything to the screen."""
        self.screen.fill(WHITE)
        
        # ... your existing drawing code ...
        
        if self.is_streamlit:
            # Convert Pygame surface to image for Streamlit
            return pygame.surfarray.array3d(self.screen).swapaxes(0,1)
        else:
            # Normal Pygame display
            pygame.display.flip()

# Main execution
if __name__ == "__main__":
    demo = DelaunayDemo()
    
    if 'streamlit' in sys.modules:
        # Streamlit interface
        st.title("Planar Pathfinding Demo")
        
        # Add Streamlit controls
        if st.button("Place Polygon"):
            demo.building_polygon = True
        if st.button("Set Start"):
            demo.placing_start = True
        if st.button("Set End"):
            demo.placing_end = True
            
        # Display the Pygame surface
        frame = demo.draw()
        st.image(frame)
    else:
        # PyScript/Pygbag interface
        async def main():
            await demo.run()
        
        asyncio.run(main()) 
