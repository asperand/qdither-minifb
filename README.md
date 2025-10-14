A re-write of qdither that uses MiniFB rather than Macroquad.

Meant to be a "lightweight" version of the original app with reduced functionality and increased performance.

The best part about this re-write is that it modifies and saves the image as soon as possible, rather than making the user wait for the visual to finish.

Despite this, the overall code could be improved in many ways. In particular, we don't necessarily need the RGB crate. We could just as easily create the 32bit buffer from the start and modify it directly. It would increase the complexity of the code significantly, but be a much leaner solution.
