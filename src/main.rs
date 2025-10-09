use minifb::{Key, ScaleMode, Window, WindowOptions};
use rgb::ComponentMap;
use image::Rgb;
use image::ImageBuffer;
use std::path::Path;
use std::io::BufRead;
use std::fs::File;
use std::vec::Vec;
use std::io;
use rgb::RGB;
use image::open;
use num_integer::div_ceil;
use clap::value_parser;
use clap::{arg, Command};
use rand::Rng;

trait RgbSatAdd {
  fn saturating_add(self, other: Self) -> Self;
}

trait RgbSatSub {
    fn saturating_sub(self, other: Self) -> Self;
}

impl RgbSatAdd for rgb::Rgb<u8> {
  fn saturating_add(self, other: Self) -> Self {
    rgb::Rgb {
      r: self.r.saturating_add(other.r),
      g: self.g.saturating_add(other.g),
      b: self.b.saturating_add(other.b),
    }
  }
}

impl RgbSatSub for rgb::RGB<u8> {
    fn saturating_sub(self, other: Self) -> Self {
        rgb::RGB {
            r: self.r.saturating_sub(other.r),
        g: self.g.saturating_sub(other.g),
        b: self.b.saturating_sub(other.b),
        }
    }
}

/// Clusters for K-Means
struct Cluster {
    centroid: RGB<u8>,
    values: Vec<RGB<u8>>,
}
/// Our arguments for the command line.
fn cli() -> Command {
    Command::new("qdither")
        .author("Ryan, ryanmc.webflow.io")
        .about("Reduces the colors in an image.")
        .version("0.1")
        .arg(arg!(<IMG> "Image file to reference."))
        .arg(arg!(<NUM> "Number of colors to reduce to.")
            .value_parser(value_parser!(u8))
            .required(false)
            .default_value("32")
            )
        .arg(arg!(<PAL> "Optional palette file")
            .required(false)
            .default_value("NONE")
            )
}

fn main() {
    let matches = cli().get_matches();
    let file_path = matches.get_one::<String>("IMG").expect("Couldn't parse image path.");
    let palette_colors = matches.get_one::<u8>("NUM").expect("Couldn't parse number of colors.");
    let palette_path = matches.get_one::<String>("PAL").expect("Couldn't parse palette path.");
    let mut image_tuple = load_file(&file_path).expect("Couldn't open file");
    let user_palette_result = setup_palette(palette_path);
    let user_palette = match user_palette_result {
        Ok(user_palette) => user_palette,
        Err(_) =>  get_colors(&mut image_tuple.0,*palette_colors), // No file specified or found? Use colors from the image.
    };
    println!("Current palette colors:");
    for color in user_palette {
      println!("#{:x}{:x}{:x}",color.r,color.g,color.b);
    }
    let image_width = image_tuple.2 as usize;
    let image_height = image_tuple.1 as usize;
    let mut buffer = vec![0u32; image_width * image_height];

    let mut window = Window::new(
        "qdither - Press Esc to stop",
        image_width,
        image_height,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to create the window");

    window.set_target_fps(60);

    let mut size = (0, 0);
    let dithered_image = dither_image_fs(&mut image_tuple.0,image_tuple.2,image_tuple.1,user_palette);
    let new_raw = to_raw_from_rgb(dithered_image);
    let new_buffer: ImageBuffer<Rgb<u8>, _> = ImageBuffer::from_raw(image_tuple.2,image_tuple.1,new_raw).unwrap();
    match new_buffer.save("./dither.png") {
        Err(_) => println!("Couldn't save image buffer"),
        Ok(_) => println!("Saved image buffer to dither.png"),
    };
    buffer = convert_rgb8_to_buf32(dithered_image);
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let new_size = window.get_size();
        if new_size != size {
            size = new_size;
            buffer.resize(size.0 * size.1, 0);
        }
        // buffer update logic goes here.
        // The idea that I have is to iterate through both the dithered image and the original to show the changes line by line, rather than having it be a "real-time" thing.
        window
            .update_with_buffer(&buffer, new_size.0, new_size.1)
            .unwrap();
    }

    
}

/// Set up a vector of RGB values from a text-based palette file.
fn setup_palette<P>(palette_path: P) -> Result<Vec<RGB<u8>>,std::io::Error>
where P: AsRef<Path>, {
    let mut new_color = Vec::new();
    let mut user_palette = Vec::new();
    let file = File::open(palette_path)?;
    let lines = io::BufReader::new(file).lines();
    for line in lines.flatten(){
        if line.chars().any(|c| c.is_ascii_hexdigit()) == false || line.len() != 6 {
            continue; // Skip the current line if it doesn't meet our standards.
        }
        let mut cur = line.clone();
        while !cur.is_empty(){ // Recursive sub-string splitting.
            let (color, rest) = cur.split_at(2);
            new_color.push(u8::from_str_radix(color,16).unwrap());
            cur = rest.to_string();
        }
        let pal_rgb = RGB {r:new_color[0], g:new_color[1], b:new_color[2]};
        user_palette.push(pal_rgb);
        new_color.clear();
    }
    if user_palette.is_empty(){ // Only fires if there were no valid colors in the file.
        user_palette.push(RGB{r:3,g:3,b:3});
        user_palette.push(RGB{r:255,g:255,b:255});
        println!("No valid colors found, default palette will be used.");
    }
    Ok(user_palette)
}

/// Load an image file from a path.
fn load_file(file_path : &String ) -> Result<(Vec<rgb::Rgb<u8>>, u32 ,u32),image::error::ImageError>{
    let mut image_rgb_vec = Vec::new();
    let image_file = open(file_path)?.to_rgb8();
    let image_height = image_file.height();
    let image_width = image_file.width(); // We need these as the raw sequence doesn't have h/w.
    let image_raw = image_file.into_raw(); // Converting DynamicImage into a raw u8 sequence.
    for i in (0..image_raw.len()).step_by(3) { // For each 3 channel groupings, put them into a Vec.
        image_rgb_vec.push(RGB{r:image_raw[i],g:image_raw[i+1],b:image_raw[i+2]});
    }
    Ok((image_rgb_vec,image_height,image_width)) // Returning vec of invididual RGB values.
}

/// Simple Euclidean distance between two pixels.
fn eu_distance_pixels(current_pixel:RGB<u8>, compared_pixel:RGB<u8>) -> f32 {
    let eu_distance = // Using color weighting
            (((current_pixel.r as f32 - compared_pixel.r as f32)* 0.3).powi(2)
            +((current_pixel.g as f32 - compared_pixel.g as f32) * 0.59).powi(2)
            +((current_pixel.b as f32 - compared_pixel.b as f32) * 0.11).powi(2))
            .sqrt();
    return eu_distance;
}

/// From a single pixel, find the closest color in a vector of colors.
fn find_nearest_color(current_color:RGB<u8>,user_palette:Vec<RGB<u8>>) -> RGB<u8> {
    let mut lowest = 0;
    let mut max_distance = 441.672956; // Max possible distance in a 256x256x256 box
    for i in 0..user_palette.len() {
        let eu_distance = // Again, using color weighting.
            (((current_color.r as f32 - user_palette[i].r as f32) * 0.3).powi(2)
            +((current_color.g as f32 - user_palette[i].g as f32) * 0.59).powi(2)
            +((current_color.b as f32 - user_palette[i].b as f32) * 0.11).powi(2))
            .sqrt();
        if eu_distance < max_distance {
            max_distance = eu_distance;
            lowest = i;
        }        
    }
    return user_palette[lowest] // Return our new color!
}

/// Edit the image file pixel by pixel, dithering it with Floyd-Steinberg Error Diffusion
fn dither_image_fs(image_rgb_vec:&mut Vec<RGB<u8>>, width:u32, height:u32, user_palette:Vec<RGB<u8>>) -> Vec<RGB<u8>> {
    let mut wrapper_left = true;
    let mut wrapper_right = false;
    let mut wrapper_end = false;
    if height == 1 { // If the image is 1 pixel tall we start at the bottom.
        wrapper_end = true;
    }
    for i in 0..(image_rgb_vec.len()-1){ // For every pixel in the image...
        let i_a = i as u32;
        let new_color = find_nearest_color(image_rgb_vec[i],user_palette.clone()); // Find nearest color in palette.
        let quant_err = image_rgb_vec[i].saturating_sub(new_color); // Quant error calculation.
        image_rgb_vec[i] = new_color;
        if !wrapper_end { // If we are not at the bottom...
            image_rgb_vec[(i_a+width) as usize].saturating_add( // [x][y+1]
                quant_err.map(|p| (p as f32 * (0.3125)).round() as u8)); // 5/16
        }
        if !wrapper_right { // If we are not at the rightmost end...
            image_rgb_vec[i+1] = image_rgb_vec[i+1].saturating_add( // [x+1],[y]
                quant_err.map(|p| (p as f32 * (0.4375)).round() as u8)); // 7/16
        }
        if !wrapper_right && !wrapper_end { // If we are either not at the rightmost end or at the bottom...
            image_rgb_vec[(i_a + (width+1)) as usize].saturating_add( // [x+1][y+1]
                quant_err.map(|p| (p as f32 * (0.0625)).round() as u8)); // 1/16
        }
        if !wrapper_left && !wrapper_end { // If we are not at the leftmost end or at the bottom...
            image_rgb_vec[(i_a + (width-1)) as usize].saturating_add( // [x-1][y+1]
                quant_err.map(|p| (p as f32 * (0.1875)).round() as u8)); // 3/16
        }
        if (i_a+1) % width == 0{ // We are at the left starting next loop.
            wrapper_left = true;
        }
        else {
            wrapper_left = false;
        }
        if (i_a+2) % width == 0{ // We are at the right starting next loop.
            wrapper_right = true;
        }
        else{
            wrapper_right = false;
        }
        if i_a+1 >= width*(height-1) { // We are at the bottom starting next loop.
            wrapper_end = true;
        }
    }
    return image_rgb_vec.to_vec()
}

/// Make a raw sequence of u8 for image output from a vector of RGB values.
fn to_raw_from_rgb(image_rgb_vec:Vec<RGB<u8>>) -> Vec<u8> {
    let mut raw_sequence = Vec::new();
    for pixel in image_rgb_vec {
        raw_sequence.push(pixel.r);
        raw_sequence.push(pixel.g);
        raw_sequence.push(pixel.b);
    }
    return raw_sequence
}

/// Use K-Means algorithm to find the mean colors within an image given X clusters
fn get_colors(image:&mut Vec<RGB<u8>>, palette_colors:u8) -> Vec<RGB<u8>> {
    let mut cluster_vec = Vec::<Cluster>::new(); // Empty vector of our clusters.
    for _i in 0..(palette_colors) { // How many colors we want out of the image decides how many clusters we create.
        let new_cluster = Cluster {
            centroid : image[rand::rng().random_range(0..image.len())], // A random pixel within the image.
            values : Vec::<RGB<u8>>::new() // An empty vector of pixel values.
        };
        cluster_vec.push(new_cluster)
   }
   let mut new_cent_vec = Vec::<RGB<u8>>::new(); // This is hacky, but we need this to be available function-wide so we can return it.
   loop {
        for pixel in image.into_iter() { // for each pixel in the image, we find the lowest distance cluster and assign it to that.
            let mut lowest_distance_index = 0; // reset the referential index for finding the best cluster each pixel...
            let mut max_distance = 441.672956; // And our max distance as well.
            for i in 0..(cluster_vec.len()){
                let eu_distance = eu_distance_pixels(*pixel,cluster_vec[i].centroid);
                if eu_distance < max_distance { // If there's a lower distance...
                    max_distance =  eu_distance;
                    lowest_distance_index = i;
                } // We found our lowest distance cluster.
            }
            cluster_vec[lowest_distance_index].values.push(*pixel);
        }
        let mut prev_cent_vec = Vec::<RGB<u8>>::new();
        new_cent_vec = Vec::<RGB<u8>>::new(); // Clear the tracking vectors for these centroid values.
        for i in 0..(cluster_vec.len()){
            let mut red_sum = 0;
            let mut green_sum = 0;
            let mut blue_sum = 0; // Resetting our sums for each subpixel.
            prev_cent_vec.push(cluster_vec[i].centroid);
            for pixel in &cluster_vec[i].values { // Sum all r,g,b values individually
                red_sum += pixel.r as u32;
                green_sum += pixel.g as u32;
                blue_sum += pixel.b as u32;
            }

            if cluster_vec[i].values.len() == 0 {
                cluster_vec[i].values.push(RGB{r:255,g:255,b:255});
            }

            cluster_vec[i].centroid.r = div_ceil(red_sum, cluster_vec[i].values.len() as u32) as u8;
            cluster_vec[i].centroid.g = div_ceil(green_sum, cluster_vec[i].values.len() as u32) as u8;
            cluster_vec[i].centroid.b = div_ceil(blue_sum, cluster_vec[i].values.len() as u32) as u8; // New centroid is set!

            new_cent_vec.push(cluster_vec[i].centroid); // Push our new one for comparison
            cluster_vec[i].values = Vec::<RGB<u8>>::new(); // Clear our values vec for re-assignment since the old values are not needed.
        }
        // We determine convergence by taking the mean of means and comparing euclidean distance.
        let mut is_converged = false;
        for i in 0..(new_cent_vec.len()){
            if eu_distance_pixels(new_cent_vec[i],prev_cent_vec[i]) > 4.0 { // TODO: custom tolerance?
                is_converged = false;
            }
            else{
                is_converged = true;
            }
        }
        if is_converged == true {
            break;
        }
   }
   return new_cent_vec;
}

fn convert_rgb8_to_buf32(image_rgb_vec:Vec<RGB<u8>>) -> Vec<u32> {
    let mut buf32 = Vec::new();
    for pixel in image_rgb_vec {
        let rgba_pixel = u32::from(pixel.r) << 16
        | u32::from(pixel.g) << 8 
        | u32::from(pixel.b);
        buf32.push(rgba_pixel);
    }
    return buf32
}
