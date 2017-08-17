minimap = {
    '$####', ...
    '.#X..', ...
    '.....'
};

% basis set of tasks
%
minimaps = {
    {'1####', ...
     '.#X..', ...
     '0...0'}; ...
     
    {'0####', ...
     '.#X..', ...
     '1...0'}; ...
     
    {'0####', ...
     '.#X..', ...
     '0...1'}; ...
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


L = createLMDP(minimap, 1);
L = solveLMDP(L);
sampleLMDP(L, L.a, minimap);
%sampleLMDP(L, L.P, minimap);

M = createMLMDP(minimaps, 1);
M = solveMLMDP(M);

% Create a MLMDP from multiple mazes;
% assumes mazes represent a valid basis task set
%
function M = createMLMDP(maps, lambda)
    M = [];
    
    for i = 1:numel(maps)
        map = maps{i};
        L = createLMDP(map, lambda);
    
        if isempty(M)
            M = L;
            M.Qb = L.qb;
        else
            assert(isequal(L.S, M.S));
            assert(isequal(L.P, M.P));
            assert(isequal(L.Pi, M.Pi));
            assert(isequal(L.Pb, M.Pb));
            assert(isequal(L.I, M.I));
            assert(isequal(L.B, M.B));
            assert(isequal(L.qi, M.qi));
            
            M.Qb = [M.Qb, L.qb];
        end
    end
    
    M.Nt = size(M.Qb, 2);
    assert(M.Nt == numel(maps));
end

% Solve an initialized MLMDP
%
function M = solveMLMDP(M)
    Z = [];
    for i = 1:M.Nt
        M.qb = M.Qb(:,i);
        L = solveLMDP(M);
        
        if isempty(Z)
            Z = L.z;
        else
            Z = [Z, L.z];
        end
    end    
    assert(size(Z, 2) == M.Nt);
    
    M.Z = Z;
end


% Initialize an unsolved LMDP from a maze
%
function L = createLMDP(map, lambda)
    state = @(x, y) (x - 1) * numel(map{1}) + (y - 1) + 1;
    
    N = numel(map) * numel(map{1}); % number of states
    S = 1 : N; % set of states {s}
    B = []; % set of boundary states
    P = zeros(N, N); % passive transitions P(s'|s)
    R = zeros(N, 1); % instantaneous reward f'n R(s)
    start = nan; % the starting state
    
    absorbing = '0123456789$'; % squares that correspond to boundary states
    agent = 'X'; % the starting square
    passable = ['.', agent, absorbing]; % squares that are correspond to passable / allowed states
    
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
            %fprintf('(%d, %d) --> %d = ''%c''\n', x, y, s, map{x}(y));
            assert(ismember(s, S));
            
            if ismember(map{x}(y), absorbing)
                if map{x}(y) == '$'
                    R(s) = 10; % $$$
                else
                    R(s) = str2num(map{x}(y)); % e.g. 9 = $9
                end
                B = [B, s]; % boundary / absorbing / terminal state
                continue;
            else
                R(s) = -1; % time is money
            end
            if ismember(map{x}(y), agent)
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
                %fprintf('      (%d, %d) --> %d = ''%c''\n', new_x, new_y, new_s, map{new_x}(new_y));
                assert(ismember(new_s, S));
                    
                % passive transition f'n P(new_s|s)
                % will normalize later
                %
                P(new_s, s) = adj(i, 3);
            end
            
            P(:, s) = P(:, s) / sum(P(:, s)); % normalize P(.|s)
        end
    end
    
    I = setdiff(S, B); % set of internal states 
    q = exp(R / lambda); % exponentiated reward f'n
    
    % return LMDP
    %
    L.N = N;
    L.S = S;
    L.Nb = numel(B); % number of boundary states
    L.B = B;
    L.Ni = numel(I); % number of internal states
    L.I = I;
    
    L.P = P;
    L.Pb = P(B, I); % P(s'|s) for s' in B and s in I
    L.Pi = P(I, I); % P(s'|s) for s' and s in I
    
    L.R = R;
    L.q = q;
    L.qb = q(B); % q(s) for s in B
    L.qi = q(I); % q(s) for s in I
    
    L.start = start;
    L.lambda = lambda;
end


% Solve an initialized LMDP
%
function L = solveLMDP(L)
    Mi = diag(L.qi);
    zb = L.qb;
    Pi = L.Pi;
    Pb = L.Pb;
    P = L.P;
    N = L.N;
    
    % find desirability f'n z
    %
    z = zeros(N, 1);
    zi = inv(eye(L.Ni) - Mi * Pi') * (Mi * Pb' * zb);
    z(L.I) = zi;
    z(L.B) = zb;
        
    % find optimal policy a*
    %
    a = zeros(N, N);
    G = @(s) sum(P(:,s) .* z);
    for s = 1:N
        if G(s) == 0
            continue;
        end
        a(:,s) = P(:,s) .* z / G(s);
    end
    
    L.z = z;
    L.a = a;
end

% sample paths from a solved LMDP
%
function sampleLMDP(L, a, map)
    s = L.start;
    r = 0;
    
    getx = @(s) floor((s - 1) / numel(map{1})) + 1;
    gety = @(s) mod(s - 1, numel(map{1})) + 1;
    
    agent = 'X';
    empty = '.';
    
    disp(map');
    while ~ismember(s, L.B)
        new_s = samplePF(a(:,s));
        
        x = getx(s);
        y = gety(s);
        new_x = getx(new_s);
        new_y = gety(new_s);
        
        map{x}(y) = empty;
        map{new_x}(new_y) = agent;
        
        disp(map');
        s = new_s;
        
        r = r + L.R(s);
    end
    fprintf('Total reward: %d\n', r);
end

% sample from a discrete probability distribution
% using the universality of the normal
% i.e. F^-1(x) ~ Unif(0, 1)
%
function i = samplePF(PF)
    CDF = cumsum(PF);
    r = rand(1);
    i = find(CDF > r);
    i = i(1);
    assert(PF(i) > 0);
end