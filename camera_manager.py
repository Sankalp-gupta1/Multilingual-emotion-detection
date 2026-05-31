# Ye pura:

# ✅ ek-time-one-camera rule handle karega

# Example:

# Sankalp ON kare:

# active = True
# user = Sankalp

# Rahul try kare:

# Camera already in use

# Sankalp OFF kare:

# active = False

# phir doosra use kar sakta hai.


# ==========================================
# CAMERA MANAGER
# Only one camera active at one time
# ==========================================

camera_status = {
    "active": False,
    "user": None
}

# ==========================================
# REQUEST CAMERA
# ==========================================

def request_camera(username):

    global camera_status

    # Agar already koi use kar raha hai
    if camera_status["active"]:

        # Same user hai
        if camera_status["user"] == username:
            return True, f"{username} already using camera"

        # Dusra user use kar raha hai
        return False, f"Camera already in use by {camera_status['user']}"

    # Camera free hai
    camera_status["active"] = True
    camera_status["user"] = username

    return True, "Camera granted"


# ==========================================
# RELEASE CAMERA
# ==========================================

def release_camera(username):

    global camera_status

    # Sirf wahi release kar sakta hai
    if camera_status["user"] == username:

        camera_status["active"] = False
        camera_status["user"] = None

        return True

    return False


# ==========================================
# GET STATUS
# ==========================================

def get_camera_status():

    return camera_status