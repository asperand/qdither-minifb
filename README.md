A re-write of qdither that uses MiniFB rather than Macroquad.

Meant to be a "lightweight" version of the original app with reduced functionality and increased performance.

Despite this, the overall code could be improved in many ways. In particular, we don't necessarily need the RGB crate. We could just as easily create the 32bit buffer from the start and modify it directly. It would increase the complexity of the code significantly, but be a much leaner solution.
