$(document).ready(function() {
    /* Video Lightbox */
    if (!!$.prototype.simpleLightboxVideo) {
        $('.video').simpleLightboxVideo();
    }

    /*ScrollUp*/
    if (!!$.prototype.scrollUp) {
        $.scrollUp();
    }

    /*Responsive Navigation*/
    $("#nav-mobile").html($("#nav-main").html());
    $("#nav-trigger span").on("click", function() {
        if ($("nav#nav-mobile ul").hasClass("expanded")) {
            $("nav#nav-mobile ul.expanded").removeClass("expanded").slideUp(250);
            $(this).removeClass("open");
        } else {
            $("nav#nav-mobile ul").addClass("expanded").slideDown(250);
            $(this).addClass("open");
        }
    });

    $("#nav-mobile").html($("#nav-main").html());
    $("#nav-mobile ul a").on("click", function() {
        if ($("nav#nav-mobile ul").hasClass("expanded")) {
            $("nav#nav-mobile ul.expanded").removeClass("expanded").slideUp(250);
            $("#nav-trigger span").removeClass("open");
        }
    });

    /* Sticky Navigation */
    if (!!$.prototype.stickyNavbar) {
        $('#header').stickyNavbar();
    }

    $('#content').waypoint(function(direction) {
        if (direction === 'down') {
            $('#header').addClass('nav-solid fadeInDown');
        } else {
            $('#header').removeClass('nav-solid fadeInDown');
        }
    });

});


/* Preloader and animations */
$(window).load(function() { // makes sure the whole site is loaded
    $('#status').fadeOut(); // will first fade out the loading animation
    $('#preloader').delay(350).fadeOut('slow'); // will fade out the white DIV that covers the website.
    $('body').delay(450).css({ 'overflow-y': 'visible' });

    /* WOW Elements */
    if (typeof WOW == 'function') {
        new WOW().init();
    }

    /* Parallax Effects */
    if (!!$.prototype.enllax) {
        $(window).enllax();
    }

});
document.addEventListener('DOMContentLoaded', function() {
    var canvasContainer = document.getElementById('canvas-container');
    var canvas = new fabric.Canvas('c3', { width: canvasContainer.offsetWidth, height: canvasContainer.offsetHeight });

    var image = document.getElementById('my-image');
    var originalScale;

    function resizeCanvas() {
        canvas.setWidth(canvasContainer.offsetWidth);
        canvas.setHeight(canvasContainer.offsetHeight);

        if (canvas.item(0)) {
            var fabricImg = canvas.item(0);
            var scaleX = canvas.width / fabricImg.width;
            var scaleY = canvas.height / fabricImg.height;
            var minScale = Math.min(scaleX, scaleY);

            originalScale = minScale;

            fabricImg.set({
                left: canvas.width / 2 - (fabricImg.width * minScale) / 2,
                top: canvas.height / 2 - (fabricImg.height * minScale) / 2,
                scaleX: minScale,
                scaleY: minScale
            });

            canvas.renderAll();
        }
    }

    fabric.Image.fromURL(image.src, function(fabricImg) {
        var scaleX = canvas.width / fabricImg.width;
        var scaleY = canvas.height / fabricImg.height;
        var minScale = Math.min(scaleX, scaleY);

        originalScale = minScale;

        fabricImg.set({
            left: canvas.width / 2 - (fabricImg.width * minScale) / 2,
            top: canvas.height / 2 - (fabricImg.height * minScale) / 2,
            scaleX: minScale,
            scaleY: minScale
        });

        canvas.add(fabricImg);
    });

    window.addEventListener('resize', function() {
        resizeCanvas();
    });

    canvas.on('mouse:down', function(options) {
        if (options.target) {
            options.target.set({
                opacity: 0.5,
                borderColor: 'red',
                cornerColor: 'green',
                cornerSize: 6,
                transparentCorners: false
            });
            canvas.renderAll();
        }
    });

    canvas.on('mouse:up', function(options) {
        if (options.target) {
            options.target.set({
                opacity: 1,
                borderColor: null,
                cornerColor: null
            });
            canvas.renderAll();
        }
    });

    document.getElementById('reset-button').addEventListener('click', function() {
        if (canvas.item(0)) {
            canvas.item(0).set({
                scaleX: originalScale,
                scaleY: originalScale
            });
            canvas.renderAll();
        }
    });

    canvas.on('object:moving', function(options) {
        // Handle object moving (dragging)
    });

    canvas.on('object:scaling', function(options) {
        // Handle object scaling
    });

    // Initial canvas resize
    resizeCanvas();
});