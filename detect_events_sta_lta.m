function ev = detect_events_sta_lta(x, cfg, fs)
    staN = max(1, round(cfg.sta_win * fs));
    ltaN = max(1, round(cfg.lta_win * fs));
    sta = movmean(x.^2, staN);
    lta = movmean(x.^2, ltaN) + eps;
    r = sta ./ lta;

    thr_on = cfg.sta_on;
    thr_off = cfg.sta_off;

    idx = find(r > thr_on);
    if isempty(idx)
        ev = zeros(0, 2);
        return;
    end

    gaps = find(diff(idx) > 1);
    starts = [idx(1); idx(gaps + 1)];
    ends = [idx(gaps); idx(end)];
    res = [];

    for i = 1:numel(starts)
        s = starts(i);
        e = ends(i);

        while s > 1 && r(s) > thr_off
            s = s - 1;
        end
        while e < numel(r) && r(e) > thr_off
            e = e + 1;
        end

        if (e - s) / fs >= cfg.min_event_len
            res = [res; s, e];
        end
    end

    ev = res / fs;
end
