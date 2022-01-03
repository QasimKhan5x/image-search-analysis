const send = document.querySelector('#send')
const form = document.querySelector('#myform')
const show = document.querySelector('#show')
const loadmore = document.querySelector('#loadmore')
const loader = document.querySelector('#loader')
const fileToUpload = document.querySelector('.inputfile')
const to_be_upploaded = document.querySelector('#to-be-uploaded')

let page = 0;
let results = new Set();
fileToUpload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const src = URL.createObjectURL(e.target.files[0]);
        to_be_upploaded.src = src;
    }
    else {
        to_be_upploaded.src = ''
    }
})
const load_images = async (old=true) => {
    loader.classList.remove('hidden')
    loadmore.classList.add('hidden')
    page = page + 1
    const body = new FormData(form)
    body.append('resultLimit', page * 20)
    body.append('old',old)
    const res = await fetch('/uploadFile', {
        method: 'POST',
        body
    })
    const images = await res.json();
    loader.classList.add('hidden')
    loadmore.classList.remove('hidden')
    let newImages = new Set(images)

    const images_to_show = new Set([...newImages].filter(x => !results.has(x)))
    console.log({ newImages, images_to_show, results })
    results = union(results, images_to_show)
    images_to_show.forEach(i => {
        const div = document.createElement('div')
        div.className = 'img-box'
        div.innerHTML = `<img src="/images/${i}" alt='images ${i}' />`
        show.appendChild(div)
    });
    loadmore.classList.remove('hidden')
}
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    show.innerHTML = '';
    loadmore.classList.add('hidden')
    page = 0;
    load_images(false)
});
loadmore.addEventListener('click', ()=>{load_images(true)})

function union(setA, setB) {
    let _union = new Set(setA)
    for (let elem of setB) {
        _union.add(elem)
    }
    return _union
}
