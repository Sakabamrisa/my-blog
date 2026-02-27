const fs = require('fs');

async function fetchMap() {
    const res = await fetch('https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json');
    const json = await res.json();
    fs.writeFileSync('./src/data/china.json', JSON.stringify(json));
    console.log('Map downloaded successfully.');
}

fetchMap();
