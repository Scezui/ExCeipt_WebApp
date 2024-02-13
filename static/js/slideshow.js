// Function to load images from a folder
function loadImagesFromFolder(folder) {
    return new Promise((resolve, reject) => {
        fetch(folder)
            .then(response => response.text())
            .then(text => {
                const parser = new DOMParser();
                const htmlDoc = parser.parseFromString(text, 'text/html');
                const images = Array.from(htmlDoc.querySelectorAll('a'))
                    .filter(a => /\.(jpe?g|png|gif)$/i.test(a.href))
                    .map(a => a.href);
                resolve(images);
            })
            .catch(error => {
                reject(error);
            });
    });
}

// Function to display slideshow
function displaySlideshow(images) {
    const canvas = document.getElementById('c3');
    const ctx = canvas.getContext('2d');
    const img = document.getElementById('my-image');

    let currentIndex = 0;

    // Function to draw current image on canvas
    function drawImageOnCanvas(index) {
        img.src = images[index];
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
    }

    // Initial drawing
    drawImageOnCanvas(currentIndex);

    // Switch to the next image in the array
    function nextImage() {
        currentIndex = (currentIndex + 1) % images.length;
        drawImageOnCanvas(currentIndex);
    }

    // Start slideshow
    const slideshowInterval = setInterval(nextImage, 2000); // Change interval as needed

    // Stop slideshow after all images have been shown
    setTimeout(() => {
        clearInterval(slideshowInterval);
    }, images.length * 2000); // Adjust duration based on image count
}

// Usage
const folderPath = 'temp/uploads';
loadImagesFromFolder(folderPath)
    .then(images => {
        displaySlideshow(images);
    })
    .catch(error => {
        console.error('Error loading images:', error);
    });
