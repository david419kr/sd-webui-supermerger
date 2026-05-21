(function () {
    if (window.__superMergerLoraDropdownScrollPatch) return;
    window.__superMergerLoraDropdownScrollPatch = true;

    const dropdownSelector = "#sml_loras_dropdown";
    const itemSelector = `${dropdownSelector} .options .item`;
    const placeholderText = "Click to select LoRA";
    const preserveMs = 500;

    function appRoot() {
        if (typeof gradioApp === "function") return gradioApp();
        return document;
    }

    function setPlaceholder() {
        const input = appRoot().querySelector(`${dropdownSelector} input`);
        if (input && input.placeholder !== placeholderText) {
            input.placeholder = placeholderText;
        }
    }

    function schedulePlaceholder() {
        setPlaceholder();
        setTimeout(setPlaceholder, 100);
        setTimeout(setPlaceholder, 500);
        setTimeout(setPlaceholder, 1500);
    }

    function getOptionsList(event) {
        const target = event.target;
        if (!target || !target.closest) return null;

        const item = target.closest(itemSelector);
        if (!item) return null;

        return item.closest(".options");
    }

    function fallbackScrollTo(list, args) {
        if (args.length === 1 && typeof args[0] === "object") {
            if (typeof args[0].top === "number") list.scrollTop = args[0].top;
            if (typeof args[0].left === "number") list.scrollLeft = args[0].left;
            return;
        }

        if (typeof args[0] === "number") list.scrollLeft = args[0];
        if (typeof args[1] === "number") list.scrollTop = args[1];
    }

    function patchOptionsList(list) {
        if (!list || list.__superMergerScrollPatched) return;

        const nativeScrollTo = list.scrollTo ? list.scrollTo.bind(list) : null;
        list.scrollTo = function (...args) {
            if (Date.now() < (this.__superMergerPreserveScrollUntil || 0)) {
                if (typeof this.__superMergerPreservedScrollTop === "number") {
                    this.scrollTop = this.__superMergerPreservedScrollTop;
                }
                return;
            }

            if (nativeScrollTo) {
                nativeScrollTo(...args);
            } else {
                fallbackScrollTo(this, args);
            }
        };

        list.__superMergerScrollPatched = true;
    }

    function preserveOptionsScroll(event) {
        const list = getOptionsList(event);
        if (!list) return;

        patchOptionsList(list);

        const scrollTop = list.scrollTop;
        list.__superMergerPreservedScrollTop = scrollTop;
        list.__superMergerPreserveScrollUntil = Date.now() + preserveMs;

        const restore = () => {
            if (Date.now() <= list.__superMergerPreserveScrollUntil + 50) {
                list.scrollTop = scrollTop;
            }
        };

        if (window.queueMicrotask) queueMicrotask(restore);
        requestAnimationFrame(() => {
            restore();
            requestAnimationFrame(restore);
        });
        setTimeout(restore, 0);
        setTimeout(restore, 50);
        setTimeout(restore, 150);
        setTimeout(restore, 300);
    }

    document.addEventListener("pointerdown", preserveOptionsScroll, true);
    document.addEventListener("mousedown", preserveOptionsScroll, true);
    document.addEventListener("click", preserveOptionsScroll, true);

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", schedulePlaceholder);
    } else {
        schedulePlaceholder();
    }

    if (typeof onUiLoaded === "function") onUiLoaded(schedulePlaceholder);
    if (typeof onUiUpdate === "function") onUiUpdate(setPlaceholder);

    new MutationObserver(setPlaceholder).observe(document.documentElement, {
        childList: true,
        subtree: true,
    });
})();
