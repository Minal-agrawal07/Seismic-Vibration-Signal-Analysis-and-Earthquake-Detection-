function data = generate_synthetic(cfg)
    fs = cfg.fs;
    dur = 180;
    n = 8;        % number of files
    data = {};

    for i = 1:n
        t = (0:1/fs:dur-1/fs)';
        noise = 0.02 * randn(size(t));
        acc = noise;

        nq = randi([2, 4]);  % number of events per file
        for j = 1:nq
            len = 2 + 4 * rand();
            st = 5 + (dur - 10 - len) * rand();
            i1 = round(st * fs) + 1;
            i2 = min(numel(t), i1 + round(len * fs));

            tt = (0:(i2 - i1))' / fs;
            f0 = 3 + 6 * rand();
            wave = (1 - 2 * (pi * f0 * tt).^2) .* exp(-(pi * f0 * tt).^2);

            acc(i1:i2) = acc(i1:i2) + 0.5 * (wave + 0.3 * randn(size(wave)));
        end

        data{end + 1} = struct('filename', sprintf('synthetic_%d', i), ...
            'time', t, 'acc', acc, 'fs', fs);
    end
end
