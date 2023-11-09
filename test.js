let sortIndex = -1; // 현재 정렬되는 속성
            let sortOrder = 1; // 정렬기준) 1 : 오름차순, -1 : 내림차순
            document.querySelectorAll(".list_table tr:first-child td:not(:last-child)").forEach((element, index) => {
                element.addEventListener("click", ()=>{
                    if(sortIndex >= 0) document.querySelector(`.list_table tr:first-child td:nth-child(${sortIndex + 1})`).classList = "";
                    if(sortIndex != index) sortOrder = 1;
                    sortIndex = index;
                    element.classList = sortOrder > 0 ? 'sort_ascending' : 'sort_descending';
                    userData.sort((a, b)=>{
                        let attribute = Object.keys(a)[index];
                        let valueA = a[attribute];
                        let valueB = b[attribute];
                        if(typeof valueA === "string"){
                            if(valueA !== null && valueA !== undefined) valueA = valueA.toUpperCase();
                            if(valueB !== null && valueB !== undefined) valueB = valueB.toUpperCase();
                        }
                        if(valueA === null || valueA === undefined) return -1 * sortOrder;
                        if(valueB === null || valueB === undefined) return 1 * sortOrder;
                        if (valueA < valueB) return -1 * sortOrder;
                        if (valueA > valueB) return 1 * sortOrder;
                        return 0;
                    });
                    sortOrder*=-1;
                    createTable(userData);
                });
            });