// Coastlines and borders, fetched once per area (clipped on the server) and kept as Path2D in lon/lat units.
const cache = new Map();
export function geoLines(ext, padLon = 35, padLat = 22) {
  const r5 = (v, up) => (up ? Math.ceil(v / 5) : Math.floor(v / 5)) * 5;
  const box = [r5(ext.lon_range[0] - padLon), r5(ext.lat_range[0] - padLat), r5(ext.lon_range[1] + padLon, true), r5(ext.lat_range[1] + padLat, true)];
  const key = box.join(",");
  if (!cache.has(key)) {
    cache.set(key, fetch(`/api/geo?lon0=${box[0]}&lat0=${box[1]}&lon1=${box[2]}&lat1=${box[3]}`).then((r) => r.json()).then((g) => {
      const path = (lines) => { const p = new Path2D(); for (const l of lines) { p.moveTo(l[0], l[1]); for (let i = 2; i < l.length; i += 2) p.lineTo(l[i], l[i + 1]); } return p; };
      return { coast: path(g.coast), border: path(g.border), india: path(g.india || []) };
    }).catch(() => ({ coast: new Path2D(), border: new Path2D(), india: new Path2D() })));
  }
  return cache.get(key);
}
