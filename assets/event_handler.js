document.addEventListener('DOMContentLoaded', domReady);

function domReady() {
    new Dics({
        container: document.querySelectorAll('.b-dics')[0],
        hideTexts: false,
        textPosition: "bottom"
    });
    new Dics({
        container: document.querySelectorAll('.b-dics')[1],
        hideTexts: false,
        textPosition: "bottom"
    });
    new Dics({
        container: document.querySelectorAll('.b-dics')[2],
        hideTexts: false,
        textPosition: "bottom"
    });
}

function objectSceneEvent(dataset, idx, method_offset, div_id, dic_index) {
    let dics = document.querySelectorAll('.b-dics')[dic_index];
    let sections = dics.getElementsByClassName('b-dics__section');
    let imagesLength = 3;

    updateImages(sections, dataset, idx, method_offset, imagesLength, 'object');
    updateTabStates(div_id, idx);
}

function dropobjectSceneEvent(idx) {
    let dics = document.querySelectorAll('.b-dics')[1];
    let sections = dics.getElementsByClassName('b-dics__section');
    let columnNames = ['00', '10', '30', '50', '70', '90', '99'];

    let imageSuffix = '00019.png';
    let paths = [
        `resources/drop/3dgs/${columnNames[idx]}/${imageSuffix}`,
        `resources/drop/ours/${columnNames[idx]}/${imageSuffix}`,
        `resources/drop/gt/${columnNames[idx]}/${imageSuffix}` // Fix for GT path
    ];

    for (let i = 0; i < sections.length; i++) {
        let imageContainer = sections[i].getElementsByClassName('b-dics__image-container')[0];
        if (imageContainer) {
            let image = imageContainer.getElementsByClassName('b-dics__image')[0];
            if (image) {
                image.src = paths[i];
            }
        }
    }

    updateTabStates('dropobject-scale-recon', idx);
}

function updateImages(sections, dataset, idx, method_offset, imagesLength, sliderType) {
    for (let i = 0; i < imagesLength; i++) {
        let imageContainer = sections[i].getElementsByClassName('b-dics__image-container')[0];
        if (imageContainer) {
            let image = imageContainer.getElementsByClassName('b-dics__image')[0];
            if (image) {
                let imageFolder = getImageFolder(idx, sliderType);
                let imageFileName = getImageFileName(i + method_offset, sliderType);
                console.log("image folder: %s", imageFolder);
                console.log("image filename: %s", imageFileName);
                image.src = `./assets/imgs/${dataset}/${imageFolder}/${imageFileName}`;
            }
        }
    }
}

function updateTabStates(navId, activeIdx) {
    let navItems = document.getElementById(navId).getElementsByClassName('nav-item');
    for (let i = 0; i < navItems.length; i++) {
        navItems[i].children[0].className = (activeIdx === i) ? "nav-link active" : "nav-link";
    }
}

function getImageFolder(idx, sliderType) {
    let folders = ['Fern', 'Flower', 'Fortress', 'Horns', 'Leaves', 'Orchids', 'Room', 'Trex',
                   'Bicycle', 'Counter', 'Garden', 'Kitchen',
                   'Family', 'Francis', 'Ignatius', 'Train'];
    return folders[idx];
}

function getImageFileName(imageIndex, sliderType) {
    let filenames = ['cf_3dgs.png', '3dgs.png', 'zerogs.png'];
    return filenames[imageIndex];
}