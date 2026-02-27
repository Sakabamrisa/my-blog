export interface VisitedCity {
    name: string;
    coords: [number, number]; // [经度(longitude), 纬度(latitude)]
    postSlug: string; // 对应于 src/content/blog 中的文章的 slug，例如 "weihai-years"
}

// 在这里配置你去过的城市数据，将其与博客文章关联。点击地图上的旗帜时，将直接跳至对应的文章页面。
export const visitedCities: VisitedCity[] = [
    {
        name: "威海",
        coords: [122.122786, 37.513369], // 山东省威海市坐标
        postSlug: "weihai-years"
    }
];
