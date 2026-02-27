const fs = require('fs');

async function fetchMap() {
    try {
        const res = await fetch('https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json');
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        const json = await res.json();
        fs.writeFileSync('./src/data/china.json', JSON.stringify(json));
        console.log('Map downloaded successfully.');
    } catch (e) {
        console.error('Error fetching map:', e);
    }
}

fetchMap();
