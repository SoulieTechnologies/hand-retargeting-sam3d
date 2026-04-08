from enum import IntEnum

class HandJoint(IntEnum):
    """
    Standard 21-joint mapping for Meta's MHR/MANO hand skeleton.
    Note: This specific model structures joints from Tip -> Base.
    """
    # Thumb
    THUMB_TIP = 0
    THUMB_DIP = 1
    THUMB_PIP = 2
    THUMB_MCP = 3

    # Index Finger
    INDEX_TIP = 4
    INDEX_DIP = 5
    INDEX_PIP = 6
    INDEX_MCP = 7

    # Middle Finger
    MIDDLE_TIP = 8
    MIDDLE_DIP = 9
    MIDDLE_PIP = 10
    MIDDLE_MCP = 11

    # Ring Finger
    RING_TIP = 12
    RING_DIP = 13
    RING_PIP = 14
    RING_MCP = 15

    # Pinky Finger
    PINKY_TIP = 16
    PINKY_DIP = 17
    PINKY_PIP = 18
    PINKY_MCP = 19

    # Base
    WRIST = 20

# Groupings for calculating kinematic chains (Tip -> DIP -> PIP -> MCP -> Wrist)
FINGERS = {
    "thumb": [
        HandJoint.THUMB_TIP, 
        HandJoint.THUMB_DIP, 
        HandJoint.THUMB_PIP, 
        HandJoint.THUMB_MCP, 
        HandJoint.WRIST
    ],
    "index": [
        HandJoint.INDEX_TIP, 
        HandJoint.INDEX_DIP, 
        HandJoint.INDEX_PIP, 
        HandJoint.INDEX_MCP, 
        HandJoint.WRIST
    ],
    "middle": [
        HandJoint.MIDDLE_TIP, 
        HandJoint.MIDDLE_DIP, 
        HandJoint.MIDDLE_PIP, 
        HandJoint.MIDDLE_MCP, 
        HandJoint.WRIST
    ],
    "ring": [
        HandJoint.RING_TIP, 
        HandJoint.RING_DIP, 
        HandJoint.RING_PIP, 
        HandJoint.RING_MCP, 
        HandJoint.WRIST
    ],
    "pinky": [
        HandJoint.PINKY_TIP, 
        HandJoint.PINKY_DIP, 
        HandJoint.PINKY_PIP, 
        HandJoint.PINKY_MCP, 
        HandJoint.WRIST
    ]
}