% ==== Input ====

#defined feature/1.      % feature(F).        [F is feature]
#defined boolean/1.      % boolean(F).        [F is boolean feature]
#defined good/1.         % good(G).           [G is good transition (example)]
#defined bad/1.          % bad(B).            [B is bad transition]
#defined fixed/2.        % fixed(G, F).       [F cannot be relaxed in good transition G]
#defined source/3.       % source(T, F, V).   [T is transition and F is feature; V=0 means F is FALSE or F=0, V=1 means F is TRUE or F>0]
#defined change/3.       % change(T, F, C).   [T is transition and F is feature; C is "dec", "inc", or "eqv"]
#defined yield/1.        % yield(Y).          [Y is a joint valuation for features that may result after a good transition]
#defined yield/2.        % yield(G, Y).       [Y is a joint valuation for features that may result after good transition G]
#defined value/3.        % value(Y, F, V).    [Y is yield, F is feature; V=0 means F is FALSE or F=0, V=1 means F is TRUE or F>0]

#defined reachable/1.    % reachable(R).      [R is state reachable by sketch]
#defined transition/2.   % transition(R, T).  [T is transition at reachable state R]
#defined value_r/3.      % value_r(R, F, V).  [R is reachable state and F is feature:  V=0 means F is FALSE or F=0, V=1 means F is TRUE or F>0]
#defined force/1.        % force(R).          [Sketch must be applicable and sound at reachable state R]
#defined soft/1.         % soft(R).           [Reachable state R enters as a "soft" objective]

#defined state/2.        % state(T, S).       [S is state of type T]
#defined value_s/3.      % value_s(S, F, V).  [S is state and F is feature:  V=0 means F is FALSE or F=0, V=1 means F is TRUE or F>0]

#defined soft_constraints/0.
#defined hard_constraints/0.

% INFO: number of yields is 2**n where n is number of features
% TODO: yield/2 currently unused. It could be used to reduce grounding by removing yields that are provably impossible.


% ==== Transitions

transition(G) :- good(G).
transition(B) :- good(B).
transition(T) :- transition(R, T).


% ==== Decision variables ====

#defined condition/3.    % condition(G, F, D). [D=0 is don't care, D=1 is value of F in source of G]
#defined effect/3.       % effect(G, F, D).    [D=0 is unknown, D=1 is change of F across G]

{ condition(G, F, 0..1) } = 1 :- good(G), feature(F), not fixed(G, F).
{ effect(G, F, 0..1) } = 1    :- good(G), feature(F), not fixed(G, F).
condition(G, F, 1)            :- fixed(G, F).
effect(G, F, 1)               :- fixed(G, F).

% Equivalent transitions and equivalence classes

eq(G, G)  :- good(G).
eq(G, G2) :- good(G), good(G2), G < G2,
             condition(G2, F, D) : condition(G,  F, D);
             condition(G,  F, D) : condition(G2, F, D);
             effect(G2, F, D) : effect(G,  F, D);
             effect(G,  F, D) : effect(G2, F, D);
             source(G2, F, V) : condition(G, F, 1), source(G, F, V);
             change(G2, F, C) :    effect(G, F, 1), change(G, F, C).

class(G)  :- good(G), not eq(G2, G) : good(G2), G2 < G.


% ==== Goal and non-goal yields ====

% Goal or non-goal state S satisfies yield Y
satisfies(S, Y) :- state(_, S), yield(Y), value_s(S, F, V) : feature(F), value(Y, F, V).

% Partition yields into goal and non-goal yields
{ goal_y(Y) }   :- yield(Y).
non_goal_y(Y)   :- yield(Y), not goal_y(Y).

% State S is goal iff it satisfies some goal Y and none non-goal yield Y'
% State S is non-goal iff it satisfies some non-goal Y and none goal yield Y'
:-     state("goal", S), not satisfies(S, Y) :     goal_y(Y).
:- state("non-goal", S), not satisfies(S, Y) : non_goal_y(Y).
:-     state("goal", S), non_goal_y(Y), satisfies(S, Y).
:- state("non-goal", S),     goal_y(Y), satisfies(S, Y).


% ==== Consistent and compatible transitions ====

% Good transition G is CONSISTENT with transition T if their source states are consistent (non-symmetric relation)
consistent(G, T)    :- good(G), transition(T), source(T, F, V) : feature(F), condition(G, F, 1), source(G, F, V).

% Good transition G is COMPATIBLE with transition T if features affected by G change in T according to G (non-symmetric relation)
compatible(G, T)    :- good(G), transition(T), compatible(G, T, F) : feature(F), effect(G, F, 1).
compatible(G, T, F) :- good(G), transition(T), feature(F), effect(G, F, 1), change(G, F, C), change(T, F, C).
compatible(G, T, F) :- good(G), transition(T), boolean(F), effect(G, F, 1), change(G, F, "inc"), change(T, F, "eqv"), source(T, F, 1).
compatible(G, T, F) :- good(G), transition(T), boolean(F), effect(G, F, 1), change(G, F, "dec"), change(T, F, "eqv"), source(T, F, 0).

consistent_r(G, R)  :- good(G), reachable(R), value_r(R, F, V) : feature(F), condition(G, F, 1), source(G, F, V).
compatible_r(G, R)  :- good(G), reachable(R), transition(R, T), compatible(G, T).


% ==== Safeness ====

% Good transition G is SAFE if for each bad transition B, G is either inconsistent or incompatible with B
safe(G)    :- good(G), safe(G, B) : bad(B).
safe(G, B) :- good(G), bad(B), not consistent(G, B).
safe(G, B) :- good(G), bad(B), not compatible(G, B).


% ==== Soundness ====

% Sketch is (weakly) sound for reachable R if there is a good transition that is consistent and compatible with a transition at R, or no rule is consistent with R
sound_weak(R) :- reachable(R), not consistent_r(G, R) : good(G); not force(R).
sound_weak(R) :- reachable(R), good(G), consistent_r(G, R), compatible_r(G, R).


% ==== Structural closure ====

% Yield Y is ACTIVE if it is consistent with source of some good transition G, or it is possible after a good transition G
active(Y)         :- yield(Y), good(G), value(Y, F, V) : feature(F), source(G, F, V).
active(Y)         :- yield(Y), possible(Y, G).

% Yield Y is POSSIBLE for good transition G if it may result after (relaxation) G over an active yield Y2
possible(Y, G)    :- yield(Y), good(G), active(Y2), triggers(G, Y2),
                     value(Y, F, V) : effect(G, F, 1), change(G, F, "eqv"), value(Y2, F, V);
                     value(Y, F, 0) : effect(G, F, 1), change(G, F, "dec"), boolean(F);
                     value(Y, F, 1) : effect(G, F, 1), change(G, F, "inc").

% Good transition G TRIGGERS at yield Y if source of G is consistent with Y
triggers(G, Y)    :- good(G), yield(Y), value(Y, F, V) : feature(F), condition(G, F, 1), source(G, F, V).

% Yield Y is COVERED if it is not active, it is a goal yield, or there is good transition G that triggers at Y
covered(Y)        :- yield(Y), not active(Y).
covered(Y)        :- active(Y), goal_y(Y).
covered(Y)        :- active(Y), triggers(G, Y).

%*
% Yield Y is POSSIBLE for good transition G if it may be a result of the relaxation of G (i.e., if value for each feature F may be a result of relaxation)
possible(Y, G)    :- yield(Y), good(G), possible(Y, G, F) : feature(F).
possible(Y, G, F) :- yield(Y), good(G), feature(F), effect(G, F, 0).
possible(Y, G, F) :- yield(Y), good(G), feature(F), effect(G, F, 1), change(G, F, "dec").
possible(Y, G, F) :- yield(Y), good(G), feature(F), effect(G, F, 1), change(G, F, "inc"), value(Y, F, 1).
possible(Y, G, F) :- yield(Y), good(G), feature(F), effect(G, F, 1), change(G, F, "eqv"), value(Y, F, V), source(G, F, V).

% Good transition G is CLOSED if each of its valid yields is covered
closed(G)         :- good(G), covered(Y) : possible(Y, G).
*%


% ==== Hard/soft optimization ====

% Hard optimization (SAT)

:- good(G), not safe(G).
%:- hard_constraints, class(G), not closed(G).
:- hard_constraints, yield(Y), not covered(Y).
:- hard_constraints, reachable(R), not soft(R), not sound_weak(R).

% Soft optimization

#minimize { 1@50,      class, G     : class(G) }.
#maximize { 1@40,    closure, Y     : yield(Y), covered(Y) }.
#maximize { 1@30,  soundness, R     : reachable(R), soft(R), sound_weak(R) }.
#maximize { 1@10,    unknown, G, F  : class(G), effect(G, F, 0) }.
#maximize { 1@10,   dontcare, G, F  : class(G), condition(G, F, 0) }.


% ==== Show ====

#show.
#show eq/2.
#show class/1.
#show goal_y/1.
#show dont_care(G, F)     : class(G), condition(G, F, 0).
#show unknown(G, F)       : class(G), effect(G, F, 0).
#show non_safe(G)         : class(G), not safe(G).
%#show non_closed(G)       : class(G), not closed(G).
#show non_sound(R)        : reachable(R), not sound_weak(R).
#show non_covered(Y)      : yield(Y), not covered(Y).
#show non_active(Y)       : yield(Y), not active(Y).
#show active/1.
#show consistent_r/2.
#show compatible_r/2.
#show soft_constraints/0.
#show hard_constraints/0.

