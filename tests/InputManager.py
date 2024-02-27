from pynput import keyboard

# Manually map pynput keys to virtual key codes
# VK for Keys with char equivalents (e.g. 'a', '1', '-') can be converted using their ASCII

KEY_MAP = {
        # Keys without 'char' attribute
        keyboard.Key.space: 0x20,
        keyboard.Key.page_down: 0x22,
        keyboard.Key.left: 0x25,
        keyboard.Key.up: 0x26,
        keyboard.Key.right: 0x27,
        keyboard.Key.down: 0x28,
        keyboard.Key.insert: 0x2D,
        keyboard.Key.delete: 0x2E,
    }

class InputManager():
    def __init__(self):
        self.keystate = {} # tracks the status of every key. E.g. {Key.left: True, Key.right: False}
        self.listener = keyboard.Listener(on_press=self.onPressed, on_release=self.onReleased)

    def convertToVK(self, key: keyboard.Key | keyboard.KeyCode | str) -> int | None:
        """convertToVK
        Converts pynput Key, Keycode, and string(char) to Virtual Keys to be used as key in keystate map.

        Args:
        key: The key to convert to Virtual Key Code.

        Returns:
        The Virtual Key after converted. None if a non "char" equivalent key that has not been mapped is passed in.
        """

        if isinstance(key, keyboard.Key):
            return KEY_MAP.get(key, None)
        elif isinstance(key, keyboard.KeyCode):
            return key.vk
        else:
            return ord(key.upper())

    def onPressed(self, key: keyboard.Key | keyboard.KeyCode):
        """onPressed
        Callback function to set keystate map.

        Args:
        key: The key that has been pressed.
        """

        vk_code = self.convertToVK(key)
        if vk_code:
           self.keystate[vk_code] = True


    def onReleased(self, key):
        """onReleased
        Callback function to unset keystate map.

        Args:
        key: The key that has been released.
        """

        vk_code = self.convertToVK(key)
        if vk_code:
            self.keystate[vk_code] = False

    def start(self):
        """start
        Calls pynput listener's start(). Begin listening on a separate thread (non-blocking)
        """
        self.listener.start()

    def stop(self):
        """stop
        Calls pynput listener's stop(). Stops listening for mouse events.
        """
        self.listener.stop()
    
    def isPressed(self, key: keyboard.Key | str):
        """isPressed
        Returns true if a specific key is pressed.

        Args:
        key: A pynput keyboard.key object or a string(char) specifying the key to check.

        Returns:
        True if the key is pressed. False if not pressed or if the key is not registered
        """
        keycode = self.convertToVK(key)
        return self.keystate.get(keycode, False) if keycode else False
    
    def update(self):
        """update
        Not used for now.
        """
        pass
    