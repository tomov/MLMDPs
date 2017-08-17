minimap = {
    '$####', ...
    '.#X$.', ...
    '.....'
};

map = {
    '###################', ...
    '#.................#', ...
    '##.########...#.#.#', ...
    '#...#.....###.#...#', ...
    '###...#.#...#.##.##', ...
    '#.#.###.###.#.#...#', ...
    '#......X....#.#.#.#', ...
    '#.###########.#.#.#', ...
    '#..$.#..........#$#', ...
    '###################'
};


L = createLMDP(minimap);


function L = createLMDP(map)
    state = @(x, y) (x - 1) * numel(map{1}) + (y - 1) + 1;
    N = numel(map) * numel(map{1}); % number of states
    S = 1 : N; % set of states {s}
    B = []; % set of boundary states
    P = zeros(N, N); % passive transitions P(s'|s)
    R = zeros(N, 1); % instantaneous reward f'n R(s)
    start = nan; % the starting state
    
    absorbing = '$'; % squares that correspond to boundary states
    passable = '.$X'; % squares that are correspond to passable / allowed states
    starting = 'X'; % the starting square
    
    % adjacency list
    % each row = [dx, dy, non-normalized P(s'|s)]
    % => random walk, but also bias towards staying in 1 place
    %
    adj = [0, 0, 2; ...
        -1, 0, 1; ...
        0, -1, 1; ...
        1, 0, 1; ...
        0, 1, 1];

    % iterate over all states s
    %
    for x = 1:numel(map)
        for y = 1:numel(map{1})
            s = state(x, y);
            fprintf('(%d, %d) --> %d = ''%c''\n', x, y, s, map{x}(y));
            assert(ismember(s, S));
            
            if ismember(map{x}(y), absorbing)
                R(s) = 1; % instantaneous reward f'n
                B = [B, s]; % boundary / absorbing / terminal state
                continue;
            end
            if ismember(map{x}(y), starting)
                start = s; % starting state
            end
            if ~ismember(map{x}(y), passable)
                continue; % impassable state
            end
            
            % iterate over all adjacent states s --> s'
            %
            for i = 1:size(adj, 1)
                new_x = x + adj(i, 1);
                new_y = y + adj(i, 2);
                if new_x <= 0 || new_x > numel(map) || new_y <= 0 || new_y > numel(map{1})
                    continue % outside the map
                end
                if ~ismember(map{new_x}(new_y), passable)
                    continue; % impassable state
                end
                
                new_s = state(new_x, new_y);
                fprintf('      (%d, %d) --> %d = ''%c''\n', new_x, new_y, new_s, map{new_x}(new_y));
                assert(ismember(new_s, S));
                    
                % passive transition f'n P(new_s|s)
                % will normalize later
                %
                P(new_s, s) = adj(i, 3);
            end
            
            P(:, s) = P(:, s) / sum(P(:, s)); % normalize P(.|s)
        end
    end
    
    Nb = numel(B); % number of boundary states
    I = setdiff(S, B); % set of internal states 
    Ni = numel(I); % number of internal states
    Pb = P(B, :); % P(s'|s) for s' in B
    Pi = P(I, :); % P(s'|s) for s' in I
    
    % return LMDP
    %
    L.N = N;
    L.S = S;
    L.Nb = Nb;
    L.B = B;
    L.Ni = Ni;
    L.I = I;
    
    L.P = P;
    L.Pb = Pb;
    L.Pi = Pi;
    
    L.R = R;
    L.start = start;
end